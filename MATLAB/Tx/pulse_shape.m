function y = pulse_shape(s, sps, beta, kind)
%PULSE_SHAPE 脉冲成形：插零上采样 + 频域滤波。
%   用法: y = pulse_shape(s, sps, beta)          RRC（相干默认）
%         y = pulse_shape(s, sps, beta, kind)    kind: 'rrc' | 'rc' | 'rect'(NRZ) | 'gaussian'(beta 为 BT)
%   s 为 P×n 符号，y 为 P×(n·sps)；实输入返回实输出
if nargin < 4, kind = 'rrc'; end
if strcmp(kind, 'rect'), y = repelem(s, 1, sps); return; end
x = zeros(size(s,1), size(s,2)*sps);
x(:, 1:sps:end) = s;
f = freqaxis(size(x,2), sps);                 % 归一化到符号率
switch kind
    case {'rrc','rc'}, H = nyquist_resp(f, 1, beta, strcmp(kind,'rrc'));
    case 'gaussian',   H = exp(-(pi*f).^2 / (2*log(2)*beta^2));
end
y = fdfilter(x, H*sps);
if isreal(s), y = real(y); end
end
