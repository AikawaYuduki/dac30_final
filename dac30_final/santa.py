
import math

import numpy

from chainer import cuda
from chainer import optimizer

class Santa(optimizer.GradientMethod):

    """Santa optimization algorithm with the symmetric splitting scheme (SSS).
    See: http://arxiv.org/abs/1512.07962v1
    """
    def __init__(self, eta=0.01, sigma=0.9999, eps=0.001, C=0, gamma=0.5, delta=0.001, burnin=3000):
        self.eta = eta
        self.sigma = sigma
        self.eps = eps
        self.C = C
        self.gamma = gamma
        self.delta = delta
        self.burnin = burnin

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        state['v'] = xp.zeros_like(param.data, dtype=xp.float32)
        state['u'] = numpy.sqrt(self.eta) * xp.random.normal(size=param.data.shape).astype(xp.float32)
        state['g'] = xp.ones_like(param.data, dtype=xp.float32)
        state['alpha'] = xp.full_like(param.data, self.C, dtype=xp.float32)

    def force_not_too_small(self, x):
        return numpy.copysign(numpy.maximum(numpy.abs(x), self.delta), x)

    def update_one_cpu(self, param, state):
        v, alpha, prev_u = state['v'], state['alpha'], state['u']
        grad = param.grad
        v *= self.sigma
        v += (1 - self.sigma) * grad * grad

        g = 1 / numpy.sqrt(numpy.sqrt(v) + self.eps)
        param.data += g * prev_u / 2
        if self.t < self.burnin:
            # exploration
            inv_beta = self.gamma ** self.t
            prev_g = state['g']
            alpha += (prev_u * prev_u - self.eta * inv_beta) / 2
            u = numpy.exp(-alpha/2) * prev_u
            u += - g * grad * self.eta + numpy.sqrt(2 * prev_g * self.eta * inv_beta) * numpy.random.normal(size=param.data.shape)
            u += self.eta * inv_beta * (1 - g / self.force_not_too_small(prev_g)) / self.force_not_too_small(prev_u)
            u *= numpy.exp(-alpha/2) 
            alpha += (u * u - self.eta * inv_beta) / 2
            state['g'] = g
            state['u'] = u
        else:
            # refinement
            u = prev_u
            u *= numpy.exp(-alpha/2)
            u -= g * grad * self.eta
            u *= numpy.exp(-alpha/2)
        param.data += g * u /2



    def update_one_gpu(self, param, state):
        xp = cuda.get_array_module(param.data)
        g = xp.empty(param.data.shape, dtype=xp.float32)

        cuda.elementwise(
            'T grad, T sigma, T eps, T prev_u',
            'T data, T v, T g',
            '''v += (1 - sigma) * (grad * grad - v);
               g = 1 / sqrt(sqrt(v) + eps);
               data += g * prev_u / 2;''',
            'santa_pre')(param.grad, self.sigma, self.eps, state['u'], param.data, state['v'], g)
        if self.t < self.burnin:
            # exploration
            zeta = xp.random.normal(size=param.data.shape, dtype=xp.float32)
            inv_beta = self.gamma ** self.t
            u = xp.empty(param.data.shape, dtype=xp.float32)
            cuda.elementwise(
                'T prev_g, T prev_u, T inv_beta, T eta, T g, T zeta, T grad, T delta',
                'T alpha, T u',
                '''alpha += (prev_u * prev_u - eta * inv_beta) / 2;
                   u = exp(-alpha/2) * prev_u;
                   u += -g * grad * eta + sqrt(2 * prev_g * eta * inv_beta) * zeta;
                   T prev_g_fixed = copysign(max(abs(prev_g), delta), prev_g);
                   u += eta * inv_beta * (1 - g / prev_g_fixed) * prev_u;
                   u *= exp(-alpha/2);
                   alpha += (u * u - eta * inv_beta)/2;
                ''',
                'santa_exploration')(
                    state['g'], state['u'], inv_beta, self.eta, g, zeta, param.grad, self.delta, state['alpha'], u)
            state['g'] = g
            state['u'] = u
        else:
            # refinement
            cuda.elementwise(
                'T alpha, T g, T grad, T eta',
                'T u',
                '''u *= exp(-alpha/2);
                   u -= g * grad * eta;
                   u *= exp(-alpha/2);''',
                'santa_refinement')(
                    state['alpha'], g, param.grad, self.eta, state['u'])
        param.data += g * state['u'] / 2

class SantaE(optimizer.GradientMethod):

    """Santa optimization algorithm with the Euler scheme.
    See: http://arxiv.org/abs/1512.07962v1
    """
    def __init__(self, eta=0.01, sigma=0.9999, eps=0.001, C=0, gamma=0.5, delta=0.001, burnin=3000):
        self.eta = eta
        self.sigma = sigma
        self.eps = eps
        self.C = C
        self.gamma = gamma
        self.delta = delta
        self.burnin = burnin

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        state['v'] = xp.zeros_like(param.data, dtype=xp.float32)
        state['u'] = numpy.sqrt(self.eta) * xp.random.normal(size=param.data.shape).astype(xp.float32)
        state['g'] = xp.ones_like(param.data, dtype=xp.float32)
        state['alpha'] = xp.full_like(param.data, self.C, dtype=xp.float32)

    def force_not_too_small(self, x):
        return numpy.copysign(numpy.maximum(numpy.abs(x), self.delta), x)

    def update_one_cpu(self, param, state):
        v, alpha, prev_u = state['v'], state['alpha'], state['u']
        grad = param.grad
        v += (1 - self.sigma) * (grad * grad - v)
        g = 1 / numpy.sqrt(numpy.sqrt(v) + self.eps)
        if self.t < self.burnin:
            # exploration
            inv_beta = self.gamma ** self.t
            prev_g = state['g']
            alpha += (prev_u * prev_u - self.eta * inv_beta)
            u = numpy.exp(-alpha/2) * prev_u
            u = numpy.sqrt(2 * prev_g * self.eta * inv_beta) * numpy.random.normal(size=param.data.shape) \
                + self.eta * inv_beta * (1 - g / self.force_not_too_small(prev_g)) / self.force_not_too_small(prev_u)
            state['g'] = g
        else:
            # refinement
            u.fill(0)
        u += (1 - alpha) * prev_u - g * grad * self.eta
        state['u'] = u
        param.data += g * u



    def update_one_gpu(self, param, state):
        xp = cuda.get_array_module(param.data)
        # print state['v'][0], state['alpha'][0], state['u'][0], state['g'][0]
        g = xp.empty(param.data.shape, dtype=xp.float32)
        v, alpha = state['v'], state['alpha']
        cuda.elementwise(
            'T grad, T sigma, T eps, T prev_u',
            'T v, T g',
            '''v += (1 - sigma) * (grad * grad - v);
               g = 1 / sqrt(sqrt(v) + eps);''',
            'santa_pre')(param.grad, self.sigma, self.eps, state['u'], v, g)
        if self.t < self.burnin:
            # exploration
            zeta = xp.random.normal(size=param.data.shape, dtype=xp.float32)
            inv_beta = self.gamma ** self.t
            u = xp.empty(param.data.shape, dtype=xp.float32)
            cuda.elementwise(
                'T prev_g, T prev_u, T inv_beta, T eta, T g, T zeta, T grad, T delta',
                'T alpha, T u',
                '''alpha += (prev_u * prev_u - eta * inv_beta);
                   u = sqrt(2 * prev_g * eta * inv_beta) * zeta;
                   T prev_g_fixed = copysign(max(abs(prev_g), delta), prev_g);
                   u += eta * inv_beta * (1 - g / prev_g_fixed) * prev_u;
                ''',
                'santa_exploration')(
                    state['g'], state['u'], inv_beta, self.eta, g, zeta, param.grad, self.delta, alpha, u)
            state['g'] = g
        else:
            # refinement
            u = xp.zeros_like(param.data, dtype=xp.float32)
        cuda.elementwise(
            'T alpha, T prev_u, T eta, T g, T grad',
            'T u, T data',
            '''u += (1 - alpha) * prev_u - eta * g * grad;
               data += g * u;''',
            'santa_update')(alpha, state['u'], self.eta, g, param.grad, u, param.data)
        state['u'] = u
