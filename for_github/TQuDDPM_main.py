import argparse
import numpy as np
from keras.backend import shape
from scipy.stats import unitary_group
from opt_einsum import contract
import os

# from src.QDDPM_torch import DiffusionModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from src.QDDPM_tf import OneQubitDiffusionModel, QDDPM, MultiQubitDiffusionModel
from src.QDDPM_tf import naturalDistance, WassDistance, wass_distance
import pickle as pkl
import time
import random
import matplotlib.pyplot as plt
from matplotlib import rc

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Use GPU")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(f"error: {e}")
else:
    print("Use CPU")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)

# circle | cluster | noise | phase
parser.add_argument('--problem', type=str, default="circle")
parser.add_argument('--loss_type', type=str, default="Wasserstein")     # MMD Wasserstein

parser.add_argument('--n', type=int, default=1)
parser.add_argument('--na', type=int, default=1)

parser.add_argument('--T', type=int, default=40)
parser.add_argument('--Ndata', type=int, default=500)
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=4)


parser.add_argument('--L', type=int, default=12)

parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_t0', type=float, default=1.0)

# Ry-pi | Ry-pi | NN | NN_NN | Ry-Rz-pi | Ry-Rz-2pi
# Ry-2pi | Rx-2pi | Embedding  |
parser.add_argument('--Encode_type', type=str, default="NN")

parser.add_argument('--two_qubit_gate_type', type=str, default="CZ")
parser.add_argument('--Topology', type=str, default="star")

parser.add_argument('--L_tau', type=int, default=2)

args = parser.parse_args()

if args.Encode_type not in ["Embedding", "Ry-pi", "Rx-pi", "Ry-2pi", "Rx-2pi", "NN", "NN_NN", "Ry-Rz-pi", "Ry-pi-Rz-2pi"]:
    raise ValueError("Error in Encode_type: ", args.Encode_type)
if args.problem not in ["circle", "cluster", "noise", "phase"]:
    raise ValueError("Error in problem: ", args.problem)
if args.loss_type not in ["MMD", "Wasserstein"]:
    raise ValueError("Error in loss_type: ", args.loss_type)
if args.two_qubit_gate_type not in ["CZ", "ZZ"]:
    raise ValueError("Error in two_qubit_gate_type: ", args.two_qubit_gate_type)

tf.random.set_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

ts = time.strftime('%m%d%H%M%S', time.localtime())
# result_path = f'result/{args.problem}_{args.n}/{args.Encode_type}/s{args.seed}_L{args.L}_wt{args.weight_t0}_{ts}'    # _{ts}
# result_path = f'result/{args.problem}_{args.n}_{args.na}_{args.lr}/{args.Encode_type}_{args.two_qubit_gate_type}/s{args.seed}_L{args.L}'
result_path = f'result/{args.problem}_{args.n}_{args.na}_{args.lr}/{args.Encode_type}_{args.two_qubit_gate_type}/s{args.seed}_L{args.L}'
if args.problem == "cluster":
    result_path = f'result/{args.problem}_{args.n}_{args.na}_{args.lr}/{args.Encode_type}_{args.two_qubit_gate_type}_{args.L_tau}_{args.weight_t0}/s{args.seed}_L{args.L}'
if not os.path.exists(result_path):
    os.makedirs(result_path)  # make dir
    os.makedirs(result_path + "/generation")  # make dir

argsDict = args.__dict__  # save setting
with open(result_path + '/setting.txt', 'w') as f:
    f.writelines("Date: " + ts + '\n')
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')


def magnetization(state, n):
    M = 0
    for i in range(2**n):
        basis_M = 0.
        basis = (bin(i)[2:]).zfill(4)
        for spin in basis:
            basis_M += (np.abs(state[i])**2 * (2*int(spin)-1))
        basis_M /= n
        if np.abs(basis_M) <= 1:
            M += np.abs(basis_M)
        else:
            print("the basis mag is wrong")
    return M


def generation_circle(epoch, diffModel, model, params):
    inputs_T_tr = diffModel.HaarSampleGeneration(args.Ndata)

    # params = np.load('data/circle/QDDPMcircleYparams_n1na2T40L6_wd.npy')
    states_tr = model.backDataGeneration(inputs_T_tr, params, args.Ndata)[:, :, :2 ** args.n].numpy()
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    ys_diff = np.real(contract('tmi, ij, tmj->tm', states_diff.conj(), sy, states_diff))
    ys_train = np.real(contract('tmi, ij, tmj->tm', states_tr.conj(), sy, states_tr))
    Y2 = np.mean(ys_train ** 2, axis=1)
    training_Y2 = np.mean(ys_train ** 2, axis=1)
    T = args.T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(T + 1), np.mean(ys_diff ** 2, axis=1), 'o--', markersize=5, mfc='white', lw=2, c='r',
            zorder=5, label=r'$\rm diffusion$')

    ax.plot(range(T + 1), np.mean(ys_train ** 2, axis=1), 'o--', markersize=5, mfc='white', lw=2, c='b',
            zorder=5, label=r'$\rm training$')
    ax.fill_between(range(T + 1), np.mean(ys_train ** 2, axis=1) - np.std(ys_train ** 2, axis=1),
                    np.mean(ys_train ** 2, axis=1) + np.std(ys_train ** 2, axis=1), color='b', alpha=0.1)

    ax.hlines(1 / 3., xmin=-10, xmax=50, ls='--', lw=2, color='orange')
    ax.set_xticks(np.arange(0, T + 1, 10))
    ax.set_yticks(np.arange(0, 7, 2) * 0.1)
    ax.set_xlim(-2, 42)
    ax.set_ylim(-0.1, 0.65)
    ax.set_ylabel(r'$\overline{\langle Y\rangle^2}$', fontsize=30)
    ax.set_xlabel(r'$t$', fontsize=30)
    ax.tick_params(direction='in', length=10, width=3, top='on', right='on', labelsize=30)
    plt.legend()
    plt.savefig(result_path + f"/generation/Y2_epoch{epoch}.jpg", dpi=300)
    plt.close()
    if epoch == 0:
        np.savetxt(result_path + f'/generation/training_Y2_epoch{epoch}.txt', training_Y2, fmt='%.8f')
    np.savetxt(result_path + f'/generation/generation_Y2_epoch{epoch}.txt', Y2, fmt='%.8f')
    return Y2[0]


def generation_cluster(epoch, diffModel, model, params):
    inputs_T_tr = diffModel.HaarSampleGeneration(args.Ndata)

    states_tr = model.backDataGeneration(inputs_T_tr, params, args.Ndata)[:, :, :2 ** args.n].numpy()

    F0_test = np.abs(states_tr[:, :, 0]) ** 2
    F0_diff = np.abs(states_diff[:, :, 0]) ** 2
    T=args.T

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(T + 1), np.mean(F0_diff, axis=1), 'o--', markersize=5, mfc='white', lw=2, c='r',
            zorder=5, label=r'$\rm diffusion$')
    ax.plot(range(T + 1), 0.5 * np.ones(T + 1), '--', lw=2, c='orange')

    ax.plot(range(T + 1), np.mean(F0_test, axis=1), 'o--', markersize=5, mfc='white', lw=2, c='forestgreen',
            zorder=5, label=r'$\rm testing$')
    ax.fill_between(range(T + 1), np.mean(F0_test, axis=1) - np.std(F0_test, axis=1),
                    np.mean(F0_test, axis=1) + np.std(F0_test, axis=1), color='forestgreen', alpha=0.1)

    ax.legend(fontsize=20, framealpha=0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0, T + 1, 5))
    ax.set_xlabel(r'${\rm Diffusion\:steps}\:t$', fontsize=30)
    ax.set_ylabel(r'$\overline{F_0}$', fontsize=30)
    ax.tick_params(direction='in', length=10, width=3, top='on', right='on', labelsize=30)
    plt.savefig(result_path + f"/generation/F0_epoch{epoch}.jpg", dpi=300)
    plt.close()

    np.savetxt(result_path + f'/generation/generation_data_mean_epoch{epoch}.txt', np.mean(F0_test, axis=1), fmt='%.8f')
    np.savetxt(result_path + f'/generation/generation_data_epoch{epoch}.txt', F0_test, fmt='%.8f')
    return np.mean(F0_test, axis=1)[0]


def generation_noise(epoch, diffModel, model, params):
    inputs_T_tr = diffModel.HaarSampleGeneration(args.Ndata)

    states_tr = model.backDataGeneration(inputs_T_tr, params, args.Ndata)[:, :, :2 ** args.n].numpy()

    F0_test = np.abs(states_tr[:, :, 2]) ** 2
    F0_diff = np.abs(states_diff[:, :, 2]) ** 2
    T=args.T

    fig, ax = plt.subplots(figsize=(6, 5), sharey=True)
    ax.plot(range(T + 1), np.mean(F0_diff, axis=1), 'o--', markersize=5, mfc='white', lw=2, c='r',
            zorder=5, label=r'$\rm diffusion$')

    ax.plot(range(T + 1), np.mean(F0_test, axis=1), 'o--', markersize=5, mfc='white', lw=2,
            c='forestgreen', zorder=5, label=r'$\rm testing$')
    ax.set_ylabel(r'$\overline{F_{10}}$', fontsize=30)
    ax.set_xlabel(r'$t$', fontsize=30)
    ax.legend(fontsize=20, framealpha=0)
    ax.hlines(0.25, xmin=-2, xmax=22, ls='--', lw=2, color='orange')
    ax.set_xticks(np.arange(0, T + 1, 5))
    ax.set_yticks(np.arange(5) * 0.1)
    ax.set_xlim(-1, 21)
    ax.tick_params(direction='in', length=10, width=3, top='on', right='on', labelsize=30)
    plt.savefig(result_path + f"/generation/F0_epoch{epoch}.jpg", dpi=300)
    plt.close()

    np.savetxt(result_path + f'/generation/generation_data_mean_epoch{epoch}.txt', np.mean(F0_test, axis=1), fmt='%.8f')
    np.savetxt(result_path + f'/generation/generation_data_epoch{epoch}.txt', F0_test, fmt='%.8f')
    return np.mean(F0_test, axis=1)[0]


def generation_phase(epoch, diffModel, model, params, ms_diff):
    ms_test = np.zeros((args.T + 1, args.Ndata))
    inputs_T_tr = diffModel.HaarSampleGeneration(args.Ndata)
    states_test = model.backDataGeneration(inputs_T_tr, params, args.Ndata)

    for t in range(args.T + 1):
        for i in range(args.Ndata):
            ms_test[t, i] = magnetization(states_test[t, i], args.n)
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.linspace(0, 1, 21)

    h_test, _ = np.histogram(ms_test[0], bins=bins)
    ax.plot((bins[1:] + bins[:-1]) / 2, h_test / 1e2, 'o--', c='forestgreen', mfc='white', markersize=5,
            label=r'$\rm testing$')

    h0, _ = np.histogram(ms_diff[0], bins=bins)
    ax.plot((bins[1:] + bins[:-1]) / 2, h0 / 1e4, 'o--', c='r', mfc='white', markersize=5,
            label=r'${\rm diffusion}(t=0)$')
    hT, _ = np.histogram(ms_diff[1], bins=bins)
    ax.plot((bins[1:] + bins[:-1]) / 2, hT / 1e4, 'o--', c='orange', mfc='white', markersize=5,
            label=r'${\rm diffusion}(t=T)$')

    ax.legend(fontsize=20, framealpha=0)
    ax.set_xlabel(r'$M$', fontsize=30)
    ax.set_ylabel(r'$p(M)$', fontsize=30)
    ax.tick_params(direction='in', length=10, width=3, top='on', right='on', labelsize=30)

    plt.savefig(result_path + f"/generation/F0_epoch{epoch}.jpg", dpi=300)
    plt.close()

    np.savetxt(result_path + f'/generation/generation_data_mean_epoch{epoch}.txt', np.mean(ms_test, axis=1), fmt='%.8f')
    np.savetxt(result_path + f'/generation/generation_data_epoch{epoch}.txt', ms_test, fmt='%.8f')
    return np.mean(ms_test, axis=1)[0]


def Training_t(model, params, Ndata, epochs, diffModel):
    '''
    training for the backward PQC at step t using whole dataset
    Args:
    model: QDDPM model
    t: diffusion step
    params_tot: collection of PQC parameters for steps > t
    Ndata: number of samples in training data set
    epochs: number of iterations
    '''

    states_diff = model.states_diff
    loss_hist = []  # record of training history

    # set optimizer and learning rate decay
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    generation_Y2 = []
    time0 = time.time()
    step_list = []
    if args.problem == "phase":
        states_diff = model.states_diff
        ms_diff = np.zeros((2, 10000))
        for i in range(10000):
            ms_diff[0, i] = magnetization(states_diff[0, i], args.n)
            ms_diff[1, i] = magnetization(states_diff[-1, i], args.n)
    for step in range(epochs):
        t = np.random.choice(np.arange(1, args.T), size=args.batch_size, replace=False)
        # input_tplus1 = model.prepareInput_t(inputs_T, params, t, Ndata)  # prepare input
        # t = np.concatenate([np.zeros(shape=[1]), t], axis=0).astype(np.int32)
        indices = np.random.choice(states_diff.shape[1], size=Ndata, replace=False)
        # input_tplus1 = states_diff[t + 1][indices]
        input_tplus1 = tf.gather(states_diff[t + 1], indices, axis=1)
        indices = np.random.choice(states_diff.shape[1], size=Ndata, replace=False)
        true_data = tf.reshape(tf.gather(states_diff[t], indices, axis=1), [-1, states_diff.shape[-1]])

        t0 = np.zeros([1], dtype=np.int32)
        input_tplus1_t0 = tf.gather(states_diff[t0 + 1], np.random.choice(states_diff.shape[1], size=Ndata, replace=False), axis=1)
        true_date_t0 = tf.reshape(tf.gather(states_diff[t0], np.random.choice(states_diff.shape[1], size=Ndata, replace=False), axis=1), [-1, states_diff.shape[-1]])

        with tf.GradientTape() as tape:
            output_t = model.backwardOutput_t(input_tplus1, params, t)
            output_t0 = model.backwardOutput_t(input_tplus1_t0, params, t0)

            if args.loss_type == "Wasserstein":
                loss_t = wass_distance(output_t, true_data)
                loss_0 = wass_distance(output_t0, true_date_t0)
                loss = args.weight_t0 * loss_t + loss_0
            elif args.loss_type == "MMD":
                loss_t = naturalDistance(output_t, true_data)
                loss_0 = naturalDistance(output_t0, true_date_t0)
                loss = args.weight_t0 * loss_t + loss_0
            else:
                raise ValueError("loss_type must be either Wasserstein or MMD, get: ", args.loss_type)

        if args.Encode_type in ["NN", "NN_NN"]:
            if args.Encode_type == "NN":
                grads = tape.gradient(loss, [params, model.time_embedding.trainable_variables])
            else:
                grads = tape.gradient(loss, [params, model.time_embedding.trainable_variables,
                                             model.NN.trainable_variables])
            optimizer.apply_gradients(zip([grads[0]], [params]))
            optimizer.apply_gradients(zip(grads[1], model.time_embedding.trainable_variables))
            if args.Encode_type == "NN_NN":
                optimizer.apply_gradients(zip(grads[2], model.NN.trainable_variables))
        else:
            grads = tape.gradient(loss, [params])
            optimizer.apply_gradients(zip(grads, [params]))

        loss_hist.append(tf.stop_gradient(loss))  # record the current loss

        if step % 500 == 0 or step == epochs - 1:
            loss_value = loss_hist[-1]
            if args.problem == "circle":
                y2  = generation_circle(epoch=step, diffModel=diffModel, model=model, params=params)
            elif args.problem == "cluster":
                y2 = generation_cluster(epoch=step, diffModel=diffModel, model=model, params=params)
            elif args.problem == "noise":
                y2 = generation_noise(epoch=step, diffModel=diffModel, model=model, params=params)
            elif args.problem == "phase":
                y2 = generation_phase(epoch=step, diffModel=diffModel, model=model, params=params, ms_diff=ms_diff)
            else:
                raise ValueError("problem get: ", args.problem)
            generation_Y2.append(y2)
            step_list.append(step)
            print("Step %s, loss: %.8s, generation metric: %.8s, time elapsed: %s seconds" % (step, float(loss_value), y2, time.time() - time0))

    plt.plot(step_list, generation_Y2)
    # plt.savefig(result_path + f"/generation_Y2.jpg", dpi=300)
    np.savetxt(result_path + "/generation_Y2.txt", np.array(generation_Y2), fmt="%.8f")
    if args.Encode_type == "NN":
        return [tf.stop_gradient(params), model.time_embedding.trainable_variables], tf.squeeze(tf.stack(loss_hist))
    elif args.Encode_type == "NN_NN":
         return [tf.stop_gradient(params), model.time_embedding.trainable_variables, model.NN.trainable_variables], tf.squeeze(tf.stack(loss_hist))
    else:
        return tf.stop_gradient(params), tf.squeeze(tf.stack(loss_hist))

if args.n == 1:
    diffModel = OneQubitDiffusionModel(args.T, args.Ndata)
else:
    diffModel = MultiQubitDiffusionModel(args.n, args.T, args.Ndata)
inputs_T = diffModel.HaarSampleGeneration(args.Ndata)

# circle | cluster | noise | phase
if args.problem == "circle":
    states_diff = np.load(f'data/circle/circleYDiff_n{args.n}T{args.T}_N5000.npy')
elif args.problem == "cluster":
    states_diff = np.load(f'data/cluster/n{args.n}/cluster0Diff_n{args.n}T{args.T}_N1000.npy')
elif args.problem == "noise":
    states_diff = np.load(f'data/noise/corrNoiseDiff_n{args.n}T{args.T}_N5000.npy')
elif args.problem == "phase":
    states_diff = np.load(f'data/phase/tfimDiff_n{args.n}T{args.T}_N10000_np.npy')
else:
    raise ValueError("error here")

model = QDDPM(n=args.n, na=args.na, T=args.T, L=args.L, Encode_type=args.Encode_type,
              Topology=args.Topology, L_tau=args.L_tau, two_qubit_gate_type=args.two_qubit_gate_type)
model.set_diffusionSet(states_diff)

if args.Encode_type == "Embedding":
    num = 2 * model.n_tot * model.L + (1 + args.na) * args.L_tau
else:
    if args.two_qubit_gate_type == "CZ":
        num = 2 * model.n_tot * model.L
    else:
        num = (2 * model.n_tot + (model.n_tot - 1)) * model.L

# std = np.sqrt(1 / (4 * ((args.L - 1) * 3 + 2)))
# self.Theta = tf.Variable(tf.random.normal(shape=[int(num_theta)], stddev=std, dtype=tf.float32))

# params = tf.Variable(tf.random.uniform(shape=[int(num)],minval=-np.pi, maxval=np.pi, dtype=tf.float32))
#
params = tf.Variable(tf.random.normal(shape=[int(num)], dtype=tf.float32))

params, loss_hist = Training_t(model, params, args.Ndata, args.epochs, diffModel)
if args.Encode_type in ["NN", "NN_NN"]:
    with open(result_path + "/QDDPMcircleYparams.pkl", "wb") as f:
        pkl.dump(params, f)
else:
    np.save(result_path + '/QDDPMcircleYparams.npy', params.numpy())
np.savetxt(result_path + '/QDDPMcircleYlosshist.txt' , loss_hist.numpy(), fmt="%.8f")
plt.plot(loss_hist)
plt.savefig(result_path + "/loss.jpg", dpi=300)
plt.close()


