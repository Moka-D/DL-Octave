clear all;

params.a = 1;
params.b = 2;
grads.a = 3;
grads.b = 4;

lr = 0.01;
optim = optimizer.SGD(lr);
actual = optim.update(params, grads);

expected.a = params.a - grads.a .* lr;
expected.b = params.b - grads.b .* lr;

assert((actual.a == expected.a) && (actual.b == expected.b));
