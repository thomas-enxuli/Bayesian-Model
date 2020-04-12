import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
from data_utils import load_dataset
import tqdm
import pandas as pd


__author__ = 'En Xu Li (Thomas)'
__date__ = 'April 10, 2020'

def _compute_accuracy(y_test, y_estimates):
    return (y_estimates == y_test).sum() / len(y_test)

def _generate_X(x_data):
    X = np.ones((len(x_data), len(x_data[0]) + 1))
    X[:, 1:] = x_data
    return X

def _sigmoid(z):
    return 1/ (1 + np.exp(-1*z))

def _log_prior(w, sigma):
    return -len(w)/2 * np.log(2 * np.pi) - len(w)/2 * np.log(sigma) - 1/(2 * sigma) * np.dot(w.T, w)


def _prior_grad(w, sigma):
    return -1/sigma * w


def _prior_hess(w, sigma):
    return -1/sigma * np.eye(len(w))


def _log_likelihood(x, y):
    return (y.T @ np.log(_sigmoid(x))) + ((1-y).T @ np.log(1-_sigmoid(x)))#/len(y)


def _likelihood_grad(X, x_prod, y):
    grad = np.zeros(np.shape(X[0]))
    for i in range(len(x_prod)):
        grad += (y[i] - _sigmoid(x_prod[i])) * X[i]
    return grad#/len(y)

def _likelihood_hess(X, x_prod):
    hess = np.zeros((len(X[0]), len(X[0])))
    temp = np.multiply(_sigmoid(x_prod), _sigmoid(x_prod) - 1)
    for i in range(len(x_prod)):
        hess = hess + (temp[i] * np.outer(X[i], X[i].T))
    return hess#/len(x_prod)



def _log_g(hessian):
    return 1/2 * np.log(np.linalg.det(-1 * hessian)) - len(hessian) / 2 * np.log(2 * np.pi)


def _likelihood(x, y):
    likelihood = 1
    for i in range(len(x)):
        likelihood *= (sigmoid(x[i]) ** y[i]) * ((1 - sigmoid(x[i])) ** (1 - y[i]))
    return likelihood


def _prior_likelihood(w, variance):
    prior = 1
    for i in range(len(w)):
        prior *= 1 / np.sqrt(2 * np.pi * variance) * np.exp(-(w[i] ** 2) / (2 * variance))
    return prior



def run_Q1a(dataset='iris', lr=0.001):
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]

    x_train, x_test = np.vstack((x_train, x_valid)), x_test
    y_train, y_test = np.vstack((y_train, y_valid)), y_test
    var_list = [0.5, 1, 2]

    X_train = _generate_X(x_train)
    X_test = _generate_X(x_test)

    marginal_likelihoods, rval_w = {} , None

    for variance in var_list:

        w = np.zeros(np.shape(X_train[0]))

        x_prod = np.reshape(X_train @ w, np.shape(y_train))
        posterior_grad = _likelihood_grad(X_train, x_prod, y_train) + _prior_grad(w, variance)

        while 1:
            if max(posterior_grad) < 10**(-2): break
            x_prod = X_train @ w
            posterior_grad = _likelihood_grad(X_train, x_prod, y_train) + _prior_grad(w, variance)
            w = w + (lr*posterior_grad)
        hessian = _likelihood_hess(X_train, x_prod) + _prior_hess(w, variance)

        marginal_likelihoods[variance] = _log_likelihood(x_prod, y_train) + _log_prior(w, variance) - _log_g(hessian)
        if variance==1: rval_w = w

    print(marginal_likelihoods)
    print(rval_w)
    return marginal_likelihoods, rval_w







def _prob_likelihood(x, y):
    prob = 1
    for i in range(len(x)):
        prob *= (_sigmoid(x[i]) ** y[i]) * ((1 - _sigmoid(x[i])) ** (1 - y[i]))
    return prob




def _proposal_likelihood(w, proposal_var, mean):
    proposal = 1
    for i in range(len(w)):
        proposal *= 1 / math.sqrt(2 * math.pi * proposal_var) * math.exp(-((mean[i] - w[i]) ** 2) / (2 * proposal_var))
    return proposal


def _r(x, y, w, prior_var, proposal_var, mean):
    return _prob_likelihood(x, y) * _prior_likelihood(w, prior_var) / _proposal_likelihood(w, proposal_var, mean)


def _proposal(mean, variance):
    return np.random.multivariate_normal(mean=mean, cov=np.eye(np.shape(mean)[0]) * variance)


def _sample_w(sample_size, mean, variance):
    w = []
    for i in range(sample_size):
        w += [_proposal(mean, variance)]
    return w


def _compute_log_likelihood(y_pred, y):
    log_p = (y.T @ np.log(y_pred)) + ((1-y).T @ np.log(1-y_pred))
    return log_p




def _importance_sampling_train_val(mean, variances, sample_sizes, dataset='iris'):
    valid_ll, valid_acc = {}, {}

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]

    prior_variance = 1


    y_train = np.asarray(y_train, int)
    y_valid = np.asarray(y_valid, int)
    y_test = np.asarray(y_test, int)
    X_train = _generate_X(x_train)
    X_valid = _generate_X(x_valid)
    X_test = _generate_X(x_test)

    min_ll = np.inf
    for sample_size in sample_sizes:
        name = 'ss_'+str(sample_size)
        temp_ll, temp_acc = [], []
        for proposal_variance in variances:

            valid_pred = np.zeros(np.shape(y_valid))
            valid_discrete_pred = np.zeros(np.shape(y_valid))

            w = _sample_w(sample_size, mean, proposal_variance)

            bar = tqdm.tqdm(total=len(X_valid), desc=name+'_var_'+str(proposal_variance))
            for d in range(len(X_valid)):
                bar.update(1)

                r_sum = 0
                for j in range(sample_size):
                    r_sum += _r(X_train @ w[j], y_train, w[j], prior_variance, proposal_variance, mean)

                pred_sum = 0
                for i in range(sample_size):
                    y_star = _sigmoid(X_valid[d] @ w[i])
                    pred_sum += y_star*_r((X_train @ w[i]), y_train, w[i], prior_variance, proposal_variance, mean)/r_sum


                valid_pred[d] = pred_sum
                valid_discrete_pred[d] = (pred_sum > 0.5)

            valid_log_likelihood = -_compute_log_likelihood(valid_pred, y_valid)/len(y_valid)
            cur_valid_acc = _compute_accuracy(valid_discrete_pred, y_valid)
            temp_ll += [valid_log_likelihood]
            temp_acc += [cur_valid_acc]
            if valid_log_likelihood < min_ll:
                min_ll = valid_log_likelihood
                min_acc = cur_valid_acc
                opt_var = proposal_variance
                opt_ss = sample_size

        valid_ll[name] = temp_ll
        valid_acc[name] = temp_acc

    df = pd.DataFrame(valid_ll, index =variances)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    df.to_csv ('validation_ll.csv')

    df = pd.DataFrame(valid_acc, index =variances)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    df.to_csv ('validation_acc.csv')

    return opt_var, opt_ss, min_ll, min_acc


def _importance_sampling_test(mean, ss, var, dataset='iris'):
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]

    x_train = np.vstack((x_train, x_valid))
    X_train = _generate_X(x_train)
    y_train = np.vstack((y_train, y_valid))
    X_test = _generate_X(x_test)

    prior_var = 1
    test_pred = np.zeros(np.shape(y_test))
    test_discrete_pred = np.zeros(np.shape(y_test))

    w = _sample_w(ss, mean, var)

    bar = tqdm.tqdm(total=len(X_test), desc='test')
    for d in range(len(X_test)):
        bar.update(1)
        r_sum = 0
        for j in range(ss):
            r_sum += _r((X_train @ w[j]), y_train, w[j], prior_var, var, mean)


        pred_sum = 0
        for i in range(ss):
            y_star = _sigmoid(X_test[d] @ w[i])
            pred_sum += y_star*_r((X_train @ w[i]), y_train, w[i], prior_var, var, mean)/r_sum

        prediction = pred_sum
        test_pred[d] = prediction

        test_discrete_pred[d] = (prediction > 0.5)


    test_acc = _compute_accuracy(test_discrete_pred, y_test)
    test_ll = _compute_log_likelihood(test_pred, y_test)/len(y_test)
    print(test_ll)
    print(test_acc)

    #visualize

    r_sum = 0
    for j in range(ss):
        r_sum += _r((X_train @ w[j]), y_train, w[j], prior_var, var, mean)

    posterior = []
    for i in range(ss):
        posterior.append(_r((X_train @ w[i]), y_train, w[i], prior_var, var, mean) / r_sum)

    for i in range(len(w[0])):
        weights = [w[j][i] for j in range(len(w))]
        weights, posterior = zip(*sorted(zip(weights, posterior)))

        z = np.polyfit(weights, posterior, 1)
        z = np.squeeze(z)
        p = np.poly1d(z)

        w_all = np.arange(min(weights), max(weights), 0.001)
        q_w = scipy.stats.norm.pdf(w_all, mean[i], var)
        plt.figure(i)
        plt.title("Posterior vis: q(w) mean=" + str(round(mean[i], 2)) + " var=" + str(var))
        plt.xlabel("w[" + str(i) + "]")
        plt.ylabel("Probability")
        plt.plot(w_all, q_w, '-b', label="Proposal q(w)")
        plt.plot(weights, posterior, 'or', label="Posterior P(w|X,y)")
        plt.plot(weights, p(weights),"r--")
        plt.legend(loc='upper right')
        plt.savefig("weight_vis_" + str(i) + ".png")


    return test_ll, test_acc

def run_Q1b():
    w_mean = [-0.87798275,  0.2951767,  -1.2357531,   0.67146419, -0.88960548]
    sample_sizes = [10, 50, 100, 500, 1000]
    variances = [0.5, 1, 2, 5]

    opt_var, opt_ss, min_ll, min_acc = \
        _importance_sampling_train_val(mean=w_mean, variances=variances, sample_sizes=sample_sizes)

    test_ll, test_acc  = _importance_sampling_test(mean=w_mean, ss=opt_ss, var=opt_var, dataset='iris')


if __name__ == '__main__':
    run_Q1a()
    run_Q1b()
