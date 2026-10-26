function p = load_params(file)
%LOAD_PARAMS 读取共用参数文件（默认 ../params/fiber_bands.json），返回结构体（bands / fibers / raman / wdm）。
%   以 '_' 开头的说明键会被 jsondecode 改名为 x_...，程序不使用。
persistent cache
if nargin < 1
    file = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'params', 'fiber_bands.json');
    if ~isempty(cache), p = cache; return; end
end
p = jsondecode(fileread(file));
if nargin < 1, cache = p; end
end
