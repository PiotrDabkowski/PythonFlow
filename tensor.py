import numpy as np
from scipy.special import expit



def broadcast_shape(shp1, shp2):
    #magic_num = 11*12+1
    try:
       # to_magic = lambda x : tuple((e if e is not None else magic_num) for e in x)
        #from_magic = lambda x: tuple((e if e!=magic_num else None) for e in x)
        return np.broadcast(np.empty(shp1), np.empty(shp2)).shape
    except ValueError:
        raise ValueError("Arrays cannot be broadcasted - %s and %s " % (str(shp1), str(shp2)))


def ensum_shape(like, grad):
    '''makes sure that the grad has the same shape as like!'''
    ls = like.shape
    gs = grad.shape
    if ls==gs:
        return grad
    to_sum = []
    dim = 0
    if len(ls)<len(gs):
        lsex = [1]*(len(gs)-len(ls)) + list(ls)
    else:
        lsex = ls
    for a, b in zip(lsex, gs):
        if a!=b:
            if a!=1:
                raise RuntimeError("Weird broadcasting - %s and %s !" % (str(ls), str(gs)))
            to_sum.append(dim)
        dim += 1
    return np.sum(grad, tuple(to_sum)).reshape(ls)

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

    @property
    def T(self):
        return transpose_op(self)



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
        elif isinstance(value, np.ndarray):
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

class sum_op(Operation):
    '''MatrixMultiplication2D'''
    def __init__(self, a, b):
        self.involvements = []

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
        if nvar==0:
            a = session.get_session_definition(self.a)
            return ensum_shape(a, my_grad)
        if nvar==1:
            b = session.get_session_definition(self.b)
            return ensum_shape(b, my_grad)
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
        a = session.get_session_definition(self.a)
        b = session.get_session_definition(self.b)
        if nvar==0:
            cand = my_grad * b
            return ensum_shape(a, cand)
        elif nvar==1:
            cand = my_grad * a
            return ensum_shape(b, cand)
        else:
            raise RuntimeError()

class mask_select_op(Operation):
    def __init__(self, a, mask):
        if a.shape!=mask.shape:
            raise ValueError("Mask must have the same shape as masked tensor!")
        self.involvements = []
        a.involvements.append((self, 0))
        # fuck the mask, not involved
        self.a = a
        self.mask = mask
        self.shape = ()

    def perform(self, session):
        return np.array(np.sum(self.a.get_value(session) * self.mask.get_value(session)))

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        mask = session.get_session_definition(self.mask)
        if nvar==0:
            return mask*my_grad
        else:
            raise RuntimeError()



# Single arg ops ----
class SingleVarOperation(Operation):
    def __init__(self, a):
        self.involvements = []
        a.involvements.append((self, 0))
        self.a = a
        self.shape = a.shape

class index_op(Operation):
    '''You can choose only a single value!!!'''
    def __init__(self, a, idx):
        try:
            np.zeros(a.shape)[idx]
        except:
            raise ValueError("Shape of indexed tensor must be known! Also index must be valid")
        self.involvements = []
        a.involvements.append((self, 0))
        self.index = idx
        self.a = a
        self.shape = ()

    def perform(self, session):
        a = self.a.get_value(session)
        return np.array( a[self.index] )

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            mask = np.zeros(session.get_session_definition(self.a).shape)
            mask[self.index] = my_grad
            return mask
        else:
            raise RuntimeError()




class exp_op(SingleVarOperation):
    def perform(self, session):
        return np.exp(self.a.get_value(session))

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            return my_grad*np.exp(self.a.get_value(session))  #todo make that faster by taking value from already calculated output!
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



class ele_sum_op(Operation):
    '''sum all elements across this dimension'''
    def __init__(self, a, dimension=0):
        if not isinstance(dimension, int):
            raise TypeError('Dimension must be an integer')
        self.involvements = []
        a.involvements.append((self, 0))
        self.a = a
        sh = list(a.shape)
        if not dimension<len(sh):
            raise ValueError('Tensor does not have this dimension - %d!' % dimension)
        sh[dimension] = 1
        self.shape = tuple(sh)
        self.dim = dimension

    def perform(self, session):
        return np.sum(self.a.get_value(session), axis=self.dim)

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            a = session.get_session_definition(self.a)
            return my_grad*np.ones_like(a)
        else:
            raise RuntimeError()




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


class transpose_op(Operation):
    def __init__(self, a):
        self.involvements = []
        a.involvements.append((self, 0))
        self.a = a
        self.shape = tuple(a.shape[::-1])


    def perform(self, session):
        return self.a.get_value(session).T

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            return my_grad.T
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




# activations...


class relu_op(SingleVarOperation):
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

class sigmoid_op(SingleVarOperation):
    def perform(self, session):
        return expit(self.a)

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            out = expit(session.get_session_definition(self.a)) #todo make that faster by taking value from already calculated output!
            return my_grad*out*(1-out)
        else:
            raise RuntimeError()


class tanh_op(SingleVarOperation):
    def perform(self, session):
        a = self.a.get_value(session)
        return np.tanh(a)

    def op_grads(self, nvar, session):
        my_grad = self.backprop(session)
        if my_grad is None:
            return None
        if nvar==0:
            out = np.tanh(session.get_session_definition(self.a))
            return my_grad*(1.0-out**2)  #todo make that faster by taking value from already calculated output!
        else:
            raise RuntimeError()


OPERATORS = {
      "__add__": sum_op,
      "__sub__": sub_op,
      "__mul__": matmul_op,
      "__div__": div_op,
      "__neg__": neg_op,
      "__getitem__": index_op
  }
for k, v in OPERATORS.items():
    def wrap(func):
        def temp(*args):
            return func(*args)
        return temp
    setattr(Tensor, k, wrap(v))

class Session:
    def __init__(self):
        self.reset()

    def get_session_definition(self, tensor):
        return self._sess_defs.get(tensor)

    def define_in_session(self, tensor, value):
        if not isinstance(tensor, Placeholder):
            raise TypeError('You can only define placeholders!')
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

    def reset(self):
        self._sess_defs = {}
        self._grads = {}





if __name__=='__main__':
    aim = np.array([[1, 0],
                    [0, 1],
                    [0, 0]])
    sess = Session()

    inp = Placeholder((3, 2))
    select = Constant(aim)
    soft = softmax_op(inp)
    loss = -mask_select_op(log_op(soft), select)

    in_arr = np.random.randn(3, 2)
    num = 0
    while num<10000:
        sess.reset()
        sess.define_in_session(inp, in_arr)
        ls = loss.get_value(sess)
        sess._grads[loss] = np.array(1)
        in_arr -= inp.backprop(sess)*0.01

        if not num%1000:
            print '-'*80
            print 'Iteration', num, '. Loss:', ls
            print soft.get_value(sess)
            print
        num+=1

    if np.sum(np.abs(soft.get_value(sess) - aim))< 0.08:
        print 'Worked!'
    else:
        print 'Something wrong with your gradients...'








