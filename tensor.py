import numpy as np
np.random.seed(10)


def broadcast_shape(a, b):
    return a


def ensum_shape(like, grad):
    '''makes sure that the grad has the same shape as like!'''
    if like.shape==grad.shape:
        return grad
    if len(like.shape)!=len(grad.shape):
        raise RuntimeError('Unexpected broadcasting type')
    to_sum = []
    dim = 0
    for a, b in zip(like.shape, grad.shape):
        if a!=b:
            if a!=1:
                raise RuntimeError("Unexpected broadcasting!")
            to_sum.append(dim)
        dim += 1
    return np.sum(grad, tuple(to_sum))

class TensorUndefined(Exception):
    pass

class Tensor:
    shape = None
    value = None
    involvements = None

    def get_value(self, session):
        raise NotImplementedError('Tensor is supposed to be inherited. Use variable, placeholder constant or operartion')

    def check_if_fits(self, val, throw=False):
        return self.are_marching_shapes(self.shape, val.shape, throw)

    def backprop(self, session):
        if self in session._grads:
            return session._grads[self]
        grad = None
        for op, nvar in self.involvements:
            cand = op.op_grads(nvar, session)
            if cand is not None:
                if grad is None:
                    grad = cand
                else:
                    grad += cand
        session._grads[self] = grad
        return grad

    @staticmethod
    def are_marching_shapes(a, b, throw=False):
        if len(a)!=len(b):
            if throw:
                raise ValueError("Dimensions do not match!")
            return False
        for x, y in zip(a, b):
            if x!=y and y is not None and x is not None:
                if throw:
                    raise ValueError("Tensors have different shapes!")
                return False
        return True



##############################

class Placeholder(Tensor):
    def __init__(self, shape):
        self.shape = shape
        self.involvements = []

    def get_value(self, session):
        val = session.get_session_definition(self)
        if val is None:
            raise TensorUndefined("Tensor undefined!" )
        return val


class Variable(Tensor):
    def __init__(self, value):
        self.involvements = []
        if isinstance(value, (float, int, long)):
            val  = np.array(value)
        if isinstance(value, np.ndarray):
            val = value
        elif isinstance(value, Constant):
            val = value.value
        else:
            raise TypeError('Initial value must be a constant, np.ndarray or a number')
        self.shape = val.shape
        self.value = val

    def get_value(self, session):
        val = session.get_session_definition(self)
        if val is not None:
            return val
        session._sess_defs[self] = self.value
        return self.value


class Constant(Variable):
    pass


class Operation(Tensor):
    shape = None

    def get_value(self, session):
        val = session.get_session_definition(self)
        if val is not None:
            return val
        # not calculated yet...
        val = self.perform(session)
        session._sess_defs[self] = val
        return val

    def perform(self, session):
        # Simply return the result of this operation
        raise NotImplementedError()

    def op_grads(self, nvar, session):
        # bacpropagate SELF and later return grad of given operand!
        raise NotImplementedError()

# Operations... --------------------------

class matmul_op(Operation):
    '''MatrixMultiplication2D'''
    def __init__(self, a, b):
        self.involvements = []

        if not (len(a.shape)==len(b.shape)==2):
            raise ValueError("dimensions of tensors must match and only 2D arrays supported!")
        if a.shape[1]!=b.shape[0]:
            raise ValueError('Shapes of tensors must match in order to perform matmul')
        if not a.shape[1]:
            raise ValueError('Width of the first tensor has to be known!')
        self.a = a
        self.b = b
        a.involvements.append((self, 0))
        b.involvements.append((self, 1))
        self.shape = self.a.shape[0], self.b.shape[1]


    def perform(self, session):
        return np.matmul(self.a.get_value(session), self.b.get_value(session))


    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            b = session.get_session_definition(self.b)
            return np.matmul(my_grad, b.T)
        elif nvar==1:
            a = session.get_session_definition(self.a)
            return np.matmul(a.T, my_grad)
        else:
            raise RuntimeError()
np.broadcast

class sum_op(Operation):
    '''MatrixMultiplication2D'''
    def __init__(self, a, b):
        self.involvements = []

        a.check_if_fits(b, throw=True)

        self.a = a
        self.b = b
        a.involvements.append((self, 0))
        b.involvements.append((self, 1))
        self.shape = broadcast_shape(self.a.shape, self.b.shape)

    def perform(self, session):
        # broadcasted automatically! watch out in backprop
        return self.a.get_value(session) + self.b.get_value(session)

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0 or nvar==1:
            return my_grad
        else:
            raise RuntimeError()

def sub_op(a, b):
    return sum_op(a, neg_op(b))

def div_op(a, b):
    return element_mul_op(a, inv_op(b))

class element_mul_op(Operation):
    '''element-wise multiplication'''
    def __init__(self, a, b):
        self.involvements = []

       # a.check_if_fits(b, throw=True)4

        self.a = a
        self.b = b
        a.involvements.append((self, 0))
        b.involvements.append((self, 1))
        self.shape = broadcast_shape(self.a.shape, self.b.shape)


    def perform(self, session):
        # broadcasted automatically! watch out in backprop
        return self.a.get_value(session) * self.b.get_value(session)

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        a = self.a.get_value(session)
        b = self.b.get_value(session)
        if nvar==0:
            cand = my_grad * b
            return ensum_shape(a, cand)
        elif nvar==1:
            cand = my_grad * a
            return ensum_shape(b, cand)
        else:
            raise RuntimeError()



# Single arg ops ----
class SingleVarOperation(Operation):
    def __init__(self, a):
        self.involvements = []
        a.involvements.append((self, 0))
        self.a = a
        self.shape = a.shape

class exp_op(SingleVarOperation):
    def perform(self, session):
        return np.exp(self.a.get_value(session))

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            return my_grad*np.exp(self.a.get_value(session))
        else:
            raise RuntimeError()


class inv_op(SingleVarOperation):
    def perform(self, session):
        return 1.0/self.a.get_value(session)

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            return -my_grad/(self.a.get_value(session)**2)
        else:
            raise RuntimeError()

class log_op(SingleVarOperation):
    def perform(self, session):
        return np.log(self.a.get_value(session))

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            return my_grad/self.a.get_value(session)
        else:
            raise RuntimeError()


def softmax_op(a):
    return div_op(exp_op(a), ele_sum_op(exp_op(a)))


def sqrt_op(a):
    return element_mul_op(a, a)



def ele_sum_op(a, dimension=0):
    '''sum all elements across this dimension'''
    if a.shape[0] is None:
        raise TypeError('Sorry I don\'t know the number of elements across this dimension :(')

    if dimension==0:
        c = Constant(np.ones((1, a.shape[0])))
        return matmul_op(c, a)




class identity_op(SingleVarOperation):
    def perform(self, session):
        return +self.a.get_value(session)

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            return my_grad
        else:
            raise RuntimeError()


class neg_op(SingleVarOperation):
    def perform(self, session):
        return -self.a.get_value(session)

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            return -my_grad
        else:
            raise RuntimeError()

class relu(SingleVarOperation):
    def perform(self, session):
        a = self.a.get_value(session)
        return np.where(a>0, a, 0.0)

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            a = session.get_session_definition(self.a)
            return np.where(a>0, my_grad, 0.0)
        else:
            raise RuntimeError()




class Session:
    def __init__(self):
        self._sess_defs = {}
        self._grads = {}

    def get_session_definition(self, tensor):
        return self._sess_defs.get(tensor)

    def define_in_session(self, tensor, value):
        if isinstance(value, Tensor):
            if isinstance(value.default_value, np.ndarray):
                val = value.default_value
            else:
                raise TypeError('If defining sess default value from tensor it must have a default value')
        elif isinstance(value, np.ndarray):
            val = value
        else:
            raise TypeError("You can define a tensor as a tensor or a np.ndarray only")
        tensor.check_if_fits(val, throw=True)
        self._sess_defs[tensor] = val

    def eval(self, tensors, defs):
        pass

sess = Session()

a = Placeholder((3, 1))

b = Constant(np.array([[1], [2], [3]]))

s = softmax_op(a)
loss = neg_op(log_op(s))
norm = np.random.randn
sess.define_in_session(a, norm(3,1))

print a.get_value(sess)
print s.get_value(sess)


sess._grads[loss] = np.array([[1], [0], [0]])
print '---\n'

print a.backprop(sess)

