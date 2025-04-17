from functools import partial
from itertools import combinations
import tensorflow_addons as tfa
import ot
import numpy as np
import scipy as sp
from scipy.stats import unitary_group

import tensorflow as tf
import tensorflow.math as tfm

import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import tensorcircuit as tc

from opt_einsum import contract

K = tc.set_backend('tensorflow')
tc.set_dtype('complex64')


class OneQubitDiffusionModel():
    def __init__(self, T, Ndata):
        '''
        the diffusion quantum circuit model to scramble arbitrary set of states to Haar random states
        Args:
        n: number of qubits
        T: number of diffusion steps
        Ndata: number of samples in the dataset
        '''
        super().__init__()
        self.t = 0
        self.T = T
        self.Ndata = Ndata

    def HaarSampleGeneration(self, Ndata):
        '''
        generate random haar states,
        used as inputs in the t=T step for backward denoise
        Args:
        Ndata: number of samples in dataset
        '''

        states_T = unitary_group.rvs(dim=2, size=Ndata)[:,:,0]

        return tf.cast(tf.convert_to_tensor(states_T), dtype=tf.complex64)
    
    def scrambleCircuit_t(self, input, phis):
        '''
        obtain the state through diffusion step t
        Args:
        t: diffusion step
        input: the input quantum state
        phis: the single-qubit rotation angles in diffusion circuit
        gs: the angle of RZZ gates in diffusion circuit when n>=2
        '''
        # input, phis = params
        c = tc.Circuit(1, inputs=input)

        for s in range(self.t):
            # single qubit rotations
            c.rz(0, theta=phis[3 * s])
            c.ry(0, theta=phis[3 * s + 1])
            c.rz(0, theta=phis[3 * s + 2])

        return c.state()
    
    def set_diffusionData_t(self, t, inputs, diff_hs, seed):
        '''
        obtain the quantum data set for 1 qubit through diffusion step t
        Args:
        t: diffusion step
        inputs: the input quantum data set
        diff_hs: the hyper-parameter to control the amplitude of quantum circuit angles
        '''
        self.t = t
        diff_hs = tf.repeat(diff_hs, 3)

        # set single-qubit rotation angles
        # tf.random.set_seed(seed)
        phis = tf.random.uniform((self.Ndata, 3 * t)) * np.pi / 4. - np.pi / 8.
        phis *= diff_hs

        # states = tf.vectorized_map(partial(self.scrambleCircuit_t, t=t), (inputs, phis))
        states = K.vmap(self.scrambleCircuit_t, vectorized_argnums=(0, 1))(inputs, phis)

        return states
    

class MultiQubitDiffusionModel():
    def __init__(self, n, T, Ndata):
        '''
        the diffusion quantum circuit model to scramble arbitrary set of states to Haar random states
        Args:
        n: number of qubits
        T: number of diffusion steps
        Ndata: number of samples in the dataset
        '''
        super().__init__()
        self.n = n
        self.T = T
        self.Ndata = Ndata
    
    def HaarSampleGeneration(self, Ndata):
        '''
        generate random haar states,
        used as inputs in the t=T step for backward denoise
        Args:
        Ndata: number of samples in dataset
        '''
        # np.random.seed(seed)
        states_T = unitary_group.rvs(dim=2 ** self.n, size=Ndata)[:,:,0]

        return tf.cast(tf.convert_to_tensor(states_T), dtype=tf.complex64)
    
    def scrambleCircuit_t(self, params, t):
        '''
        obtain the state through diffusion step t
        Args:
        t: diffusion step
        input: the input quantum state
        phis: the single-qubit rotation angles in diffusion circuit
        gs: the angle of RZZ gates in diffusion circuit when n>=2
        '''
        input, phis, gs = params
        c = tc.Circuit(self.n, inputs=input)
        for s in range(t):
            # single qubit rotations
            for i in range(self.n):
                c.rz(i, theta=phis[3 * self.n * s + i])
                c.ry(i, theta=phis[3 * self.n * s + self.n + i])
                c.rz(i, theta=phis[3 * self.n * s + 2*self.n + i])

            # homogenous RZZ on every pair of qubits (n>=2)
            for i, j in combinations(range(self.n), 2):
                c.rzz(i, j, theta=gs[s] / (2 * self.n ** 0.5))

        return c.state()
    
    def set_diffusionDataMulti_t(self, t, inputs, diff_hs, seed):
        '''
        obtain the quantum data set for multiple qubit through diffusion step t
        Args:
        t: diffusion step
        inputs: the input quantum data set
        diff_hs: the hyper-parameter to control the amplitude of quantum circuit angles
        '''
        # set single-qubit rotation angles
        # tf.random.set_seed(seed)
        phis = tf.random.uniform((self.Ndata, 3 * self.n * t)) * np.pi / 4. - np.pi / 8.
        phis *= tf.repeat(diff_hs, 3 * self.n)

        # set homogenous RZZ gate angles
        gs = tf.random.uniform((self.Ndata, t)) * 0.2 + 0.4
        gs *= diff_hs
        
        states = tf.vectorized_map(partial(self.scrambleCircuit_t, t=t), (inputs, phis, gs))

        return states


def backCircuit(input, params, n_tot, L, two_qubit_gate_type):
    '''
    the backward denoise parameteric quantum circuits,
    designed following the hardware-efficient ansatz
    output is the state before measurmeents on ancillas
    Args:
    input: input quantum state of n_tot qubits
    params: the parameters of the circuit
    n_tot: number of qubits in the circuits
    L: layers of circuit
    '''
    c = tc.Circuit(n_tot, inputs=input)
    index = 0
    for l in range(L):
        for i in range(n_tot):
            # c.rx(i, theta=params[2 * n_tot * l + i])
            # c.ry(i, theta=params[2 * n_tot * l + n_tot + i])
            c.rx(i, theta=params[index])
            index += 1
            c.ry(i, theta=params[index])
            index += 1
        for i in range(n_tot // 2):
            if two_qubit_gate_type == "CZ":
                c.cz(2 * i, 2 * i + 1)
            else:
                c.exp1(2 * i, 2 * i + 1, unitary=tc.gates._zz_matrix, theta=params[index])
                index += 1
        for i in range((n_tot-1) // 2):
            if two_qubit_gate_type == "CZ":
                c.cz(2 * i + 1, 2 * i + 2)
            else:
                c.exp1(2 * i + 1, 2 * i + 2, unitary=tc.gates._zz_matrix, theta=params[index])
                index += 1
    return c.state()


def time_embedding_circuit(t, param, na, Encode_type, Topology, L_tau, T):
    c = tc.Circuit(na)

    if Encode_type == "Ry-pi":
        theta = np.pi * t / (T + 1)
        for i in range(na):
            c.ry(i, theta=theta)
    if Encode_type == "Ry-Rz-pi":
        theta = np.pi * t / (T + 1)
        for i in range(na):
            c.ry(i, theta=theta)
            c.rz(i, theta=theta)
    elif Encode_type == "Ry-pi-Rz-2pi":
        theta = np.pi * t / (T + 1)
        theta2 = np.pi * 2 * t / (T + 1)
        for i in range(na):
            c.ry(i, theta=theta)
            c.rz(i, theta=theta2)

    elif Encode_type == "Rx-pi":
        theta = np.pi * t / (T + 1)
        for i in range(na):
            c.ry(i, theta=theta)

    elif Encode_type == "Ry-2pi":
        theta = 2 * np.pi * t / (T + 1)
        for i in range(na):
            c.ry(i, theta=theta)
    elif Encode_type == "Ry-Rz-2pi":
        theta = 2 * np.pi * t / (T + 1)
        for i in range(na):
            c.ry(i, theta=theta)
            c.rz(i, theta=theta)
    elif Encode_type == "Rx-2pi":
        theta = 2 * np.pi * t / (T + 1)
        for i in range(na):
            c.rx(i, theta=theta)
    elif Encode_type == "Embedding":
        index = 0
        theta = np.pi * t / (T + 1)
        for l in range(L_tau):
            for i in range(na):
                c.ry(i, theta=theta)
            if na == 2:
                c.exp1(0, 1, unitary=tc.gates._zz_matrix, theta=param[index])
                index += 1
            elif na == 1:
                pass
            else:
                raise ValueError("embedding error")
            for i in range(na):
                c.rx(i, theta=param[index])
                index += 1
        for i in range(na):
            c.ry(i, theta=theta)
    else:
        pass    # todo: 完成剩余的

    return c.state()


def tensor_product(a, b):
    a = tf.reshape(a, [1, -1])
    b = tf.reshape(b, [1, -1])
    result = K.kron(a, b)
    return tf.reshape(result, [-1])



class QDDPM():
    def __init__(self, n, na, T, L, Encode_type=None, Topology=None, L_tau=None, two_qubit_gate_type="ZZ"):
        '''
        the QDDPM model: backward process only work on cpu
        Args:
        n: number of data qubits
        na: number of ancilla qubits
        T: number of diffusion steps
        L: layers of circuit in each backward step
        '''
        super().__init__()
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.T = T
        self.L = L
        self.Encode_type = Encode_type
        self.Topology = Topology
        self.L_tau = L_tau
        # embed the circuit to a vectorized pytorch neural network layer
        self.backCircuit_vmap = K.jit(K.vmap(partial(backCircuit, n_tot=self.n_tot, L=L,
                                                     two_qubit_gate_type=two_qubit_gate_type), vectorized_argnums=0))


        self.timeCircuit_vmap = K.jit(K.vmap(partial(time_embedding_circuit, na=self.na, L_tau=L_tau, T=self.T,
                                               Topology=self.Topology, Encode_type=self.Encode_type),
                                   vectorized_argnums=0))
        if self.Encode_type in ["NN", "NN_NN"]:

            if self.Encode_type == "NN_NN":
                self.time_embedding = tf.keras.layers.Embedding(T, output_dim=2 * (2 ** self.na))
                self.NN = tf.keras.layers.Dense(units=2**self.na, activation=tfa.layers.GELU())
            else:
                self.time_embedding = tf.keras.layers.Embedding(T, output_dim=2 ** self.na)
            self.timeCircuit_vmap = K.jit(self.NN_embedding)

        self.tensor_product_vmap = K.jit(K.vmap(tensor_product, vectorized_argnums=[0, 1]))

    def set_diffusionSet(self, states_diff):
        self.states_diff = tf.cast(tf.convert_to_tensor(states_diff), dtype=tf.complex64)

    @tf.function
    def randomMeasure(self, inputs):
        '''
        Given the inputs on both data & ancilla qubits before measurmenets,
        calculate the post-measurement state.
        The measurement and state output are calculated in parallel for data samples
        Args:
        inputs: states to be measured, first na qubit is ancilla
        '''
        n_batch = inputs.shape[0]
        m_probs = tf.abs(tf.reshape(inputs, [n_batch, 2 ** self.na, 2 ** self.n])) ** 2.0
        m_probs = tf.reduce_sum(m_probs, axis=2)
        m_res = tfp.distributions.Categorical(probs=m_probs).sample(1)
        indices = 2 ** self.n * tf.reshape(m_res, [-1, 1]) + tf.range(2 ** self.n)
        post_state = tf.gather(inputs, indices, batch_dims=1)
        
        return tf.linalg.normalize(post_state, axis=1)[0]

    def backwardOutput_t(self, inputs, params, t=None):
        '''
        Backward denoise process at step t
        Args:
        inputs: the input data set at step t
        '''
        # outputs through quantum circuits before measurement
        # if self.Encode_type == "NN":
        #     t = tf.reshape(t, [-1, 1])
        time_embedding_state = self.timeCircuit_vmap(t, params[2 * self.n_tot * self.L:])   # 如果是embedding，则用后面的部分，
        time_embedding_state = tf.repeat(time_embedding_state[:, tf.newaxis, :], repeats=inputs.shape[1], axis=1)
        time_embedding_state = tf.reshape(time_embedding_state, [-1, time_embedding_state.shape[-1]])
        inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
        input_tplus1 = self.tensor_product_vmap(time_embedding_state, inputs)
        output_full = self.backCircuit_vmap(input_tplus1, params)
        # perform measurement
        output_t = self.randomMeasure(output_full)

        return output_t

    def NN_embedding(self, t, param):
        embed = self.time_embedding(t)
        if self.Encode_type == "NN_NN":
            embed = self.NN(embed)
        norm = tf.norm(embed, ord='euclidean', axis=-1, keepdims=True)

        # 添加数值稳定性保护（防止除零错误）
        epsilon = tf.constant(1e-12, dtype=embed.dtype)
        result = embed / tf.maximum(norm, epsilon)
        sign = tf.sign(result[..., 0:1] + tf.constant(1e-9, dtype=embed.dtype))
        return result * sign

    def prepareInput_t(self, inputs_T, params_tot, t, Ndata, param=None):
        '''
        prepare the input samples for step t
        Args:
        inputs_T: the input state at the beginning of backward
        params_tot: all circuit parameters till step t+1
        '''

        time_embedding_state = self.timeCircuit_vmap(t, param)
        self.input_tplus1 = self.tensor_product_vmap(time_embedding_state, inputs_T)
        # self.input_tplus1 = tf.concat([inputs_T, tf.zeros(shape=(Ndata, 2**self.n_tot-2**self.n),
        #                                                   dtype=tf.complex64)], axis=1)
        params_tot = tf.constant(params_tot, dtype=tf.float32)
        for tt in range(self.T-1, t, -1):
            output = self.backwardOutput_t(self.input_tplus1, params_tot[tt])
            self.input_tplus1 = tf.concat([output, tf.zeros(shape=(Ndata, 2**self.n_tot-2**self.n), 
                                                            dtype=tf.complex64)], axis=1)

        return self.input_tplus1
    
    def backDataGeneration(self, input_tplus1, params, Ndata):
        '''
        generate the dataset in backward denoise process with training data set
        '''
        states = [input_tplus1]

        # input_tplus1 = tf.concat([inputs_T, tf.zeros(shape=(Ndata, 2**self.n_tot - 2**self.n),
        #                                                   dtype=tf.complex64)], axis=1)
        params = tf.cast(tf.convert_to_tensor(params), dtype=tf.float32)
        for tt in range(self.T-1, -1, -1):
            time_embedding_state = self.timeCircuit_vmap(np.ones([1, 1], dtype=np.int32) * tt, params)
            time_embedding_state = tf.repeat(time_embedding_state[tf.newaxis, :], repeats=input_tplus1.shape[0], axis=0)
            time_embedding_state = tf.reshape(time_embedding_state, [-1, time_embedding_state.shape[-1]])
            input_tplus1 = self.tensor_product_vmap(time_embedding_state, input_tplus1)
            output_full = self.backCircuit_vmap(input_tplus1, params)
            # perform measurement
            input_tplus1 = self.randomMeasure(output_full)

            states.append(input_tplus1)
        states = tf.stack(states)[::-1]
        return states


@tf.function
def naturalDistance(Set1, Set2):
    '''
        a natural measure on the distance between two sets of quantum states
        definition: 2*d - r1-r2
        d: mean of inter-distance between Set1 and Set2
        r1/r2: mean of intra-distance within Set1/Set2
    '''
    # a natural measure on the distance between two sets, according to trace distance
    r11 = 1. - tf.reduce_mean(tf.abs(contract('mi,ni->mn', tfm.conj(Set1), Set1))**2)
    r22 = 1. - tf.reduce_mean(tf.abs(contract('mi,ni->mn', tfm.conj(Set2), Set2))**2)
    r12 = 1. - tf.reduce_mean(tf.abs(contract('mi,ni->mn', tfm.conj(Set1), Set2))**2)
    return 2*r12 - r11 - r22


def WassDistance(Set1, Set2):
    '''
        calculate the Wasserstein distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
    '''
    D = 1. - tf.abs(tfm.conj(Set1) @ tf.transpose(Set2))**2.
    emt = tf.constant([], dtype=tf.float32)
    Wass_dis = ot.emd2(emt, emt, M=D)
    return Wass_dis


import tensorflow as tf


@tf.function
def wass_distance(set1, set2, reg=0.1, num_iters=100):
    """
    """

    inner_prod = tf.matmul(tf.math.conj(set1), tf.transpose(set2))

    D = 1.0 - tf.abs(inner_prod) ** 2


    n = tf.shape(set1)[0]
    a = tf.fill([n], 1.0 / tf.cast(n, tf.float32))
    b = tf.fill([n], 1.0 / tf.cast(n, tf.float32))


    K = tf.exp(-D / reg)


    u = tf.ones_like(a)
    v = tf.ones_like(b)


    for _ in tf.range(num_iters):
        u = a / (tf.matmul(K, tf.expand_dims(v, 1))[:, 0] + 1e-8)
        v = b / (tf.matmul(tf.transpose(K), tf.expand_dims(u, 1))[:, 0] + 1e-8)


    T = tf.expand_dims(u, 1) * K * tf.expand_dims(v, 0)

    wass_dist = tf.reduce_sum(T * D)

    return wass_dist


def sinkhornDistance(Set1, Set2, reg=0.005, eps=1e-4, log=False):
    '''
        calculate the Sinkhorn distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
        reg: the regularization coefficient
        log: whether to use the log-solver
    '''
    D = 1. - tf.abs(tf.math.conj(Set1) @ tf.transpose(Set2)) ** 2.
    emt = tf.constant([], dtype=tf.float32)
    if log == True:
        sh_dis = ot.sinkhorn2(emt, emt, M=D, reg=reg, stopThr=eps, method='sinkhorn_stabilized')
    else:
        sh_dis = ot.sinkhorn2(emt, emt, M=D, reg=reg, stopThr=eps)
        
    return sh_dis

