function b = prbs_gen(order, n)
%PRBS_GEN PRBS 序列（Fibonacci LFSR，多项式 x^order + x^tap + 1）。
%   用法: b = prbs_gen(15, 1e4)    order ∈ {7, 9, 11, 15, 23, 31}，输出 1×n uint8
taps = containers.Map({7,9,11,15,23,31}, {6,5,9,14,18,28});
t = taps(order);
reg = ones(1, order);
b = zeros(1, n, 'uint8');
for i = 1:n
    new = xor(reg(order), reg(t));
    reg = [new, reg(1:end-1)];
    b(i) = new;
end
end
