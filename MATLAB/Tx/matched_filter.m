function y = matched_filter(x, sps, beta)
%MATCHED_FILTER 接收端 RRC 匹配滤波（输入 sps 倍过采样，输出同采样率）。  用法: y = matched_filter(x, 2, 0.1)
y = fdfilter(x, nyquist_resp(freqaxis(size(x,2), sps), 1, beta, true));
end
