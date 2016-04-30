from tensor import *
import random

data = ' '.join(open('text.txt').read().split())

idx_to_chr = dict(enumerate(set(data)))
chr_to_idx = {v: k for k, v in idx_to_chr.items()}
vocab_s = len(chr_to_idx)


hot_text = []
for char in data:
    hot = np.zeros((vocab_s, 1))
    hot[chr_to_idx[char]] = 1
    hot_text.append(hot)
hot_aim = hot_text[1:] + [hot_text[0]]

inp_len = 32
hidden_s = 128

Wxh = Variable(np.random.rand(hidden_s, vocab_s) * 0.01)
Bh = Variable(np.random.rand(hidden_s, 1)*0.01)
Whh = Variable(np.random.rand(hidden_s, hidden_s) * 0.01)

Why = Variable(np.random.rand(vocab_s, hidden_s) * 0.01)
By = Variable(np.random.rand(vocab_s, 1) * 0.01)



inputs = []
initial_h = Placeholder((hidden_s, 1))
h_activations = [initial_h]
y_activations = []
correct = []
loss = Constant(0)

for e in xrange(inp_len):
    inputs.append(Placeholder((vocab_s, 1)))
    h_activations.append(tanh_op( Wxh*inputs[-1] + Whh*h_activations[-1] + Bh))
    y_activations.append(softmax_op( Why*h_activations[-1] + By))
    correct.append(Placeholder((vocab_s, 1)))
    this_loss = mask_select_op(-log_op(y_activations[-1]), correct[-1])
    loss = loss + this_loss

sess = Session()
num = 0
current_pos = 10000000000000000
epoch = 0
smooth_loss = None

TO_OPTIMIZE = [Wxh, Whh, Bh, Why, By]
learning_rate = 0.01





def sample_one(x=None, h=None, sess=sess):
    if x is None:
        x = random.choice(hot_text)
    if h is None:
        h = np.zeros(initial_h.shape)
    sess.reset()
    sess.define_in_session(inputs[0], x)
    sess.define_in_session(initial_h, h)

    probs = y_activations[0].get_value(sess)
    new_h = h_activations[1].get_value(sess)

    idx = np.random.choice(np.arange(vocab_s), p=probs.ravel())
    new_x = np.zeros_like(x)
    new_x[idx] = 1
    sess.reset()
    return idx_to_chr[idx], new_x, new_h

def sample_many(n):
    text = ''
    x = h = None
    for _ in xrange(n):
        char, x, h = sample_one(x, h)
        text += char
    return text



print sample_many(40)


while True:
    sess.reset()
    if current_pos>=len(data)-inp_len:
        current_pos = 0
        epoch += 1
        print 'Epoch: ', epoch

    starting_h = np.zeros(initial_h.shape)
    sess.define_in_session(initial_h, starting_h)
    for i in xrange(inp_len):
        sess.define_in_session(inputs[i], hot_text[i+current_pos])
        sess.define_in_session(correct[i], hot_aim[i+current_pos])

    ls = loss.get_value(sess)
    smooth_loss = ls if smooth_loss is None else 0.99*smooth_loss + 0.01*ls

    sess._grads[loss] = np.array(1)
    # calculates and saves all the gradients

    for var in TO_OPTIMIZE:
        var.value -= np.clip(var.backprop(sess)*learning_rate, -0.1, 0.1)


    if not num%1000:
        print '-'*80
        print 'Loss', smooth_loss
        print sample_many(100)
        print
        print

    current_pos += inp_len
    num += 1

