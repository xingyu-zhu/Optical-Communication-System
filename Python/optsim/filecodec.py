"""文件编解码：把任意现有文件变成可调制到光信号上的比特/符号帧，并在接收端还原文件。

发射：文件 -> 封装 [文件头: 魔数/版本/文件名/长度/CRC32 | 内容] -> 比特 -> LDPC 编码 -> 加扰
      -> Gray M-QAM（比特流前半送 X、后半送 Y 偏振）-> 成帧 [训练 | 导频+数据] -> RRC 成形
接收：rx_dsp 输出的对齐帧 -> 逐比特 LLR（噪声方差由训练/导频估计）-> 解扰 -> LDPC 译码 -> 校验 CRC32 -> 文件

file_tx_signal 的返回值与 system.tx_signal 格式相同，可直接接入现有的器件、链路与 rx_dsp：
>>> from optsim.system import Config, rx_dsp
>>> from optsim import filecodec
>>> cfg = Config(M=16, rs=40e9, n_spans=10, p_launch=1)
>>> ref = filecodec.file_tx_signal('photo.png', cfg)        # 代替 tx_signal(cfg)
>>> ...                                                     # DAC / 激光器 / DP-IQ 调制器 / 链路 / 相干接收（同 examples/coherent_dp_qam.py）
>>> res = rx_dsp(adc_output, cfg, ref)
>>> out = filecodec.decode_file(res['y'], ref, out_dir='results')   # -> out['crc_ok'], out['path'] ...
"""
import os
import struct
import zlib
import numpy as np
from . import tx, fec
from .utils import normalize

MAGIC, VERSION = b'OPTF', 1
HDR = struct.Struct('>4sBHQI')                      # 魔数, 版本, 文件名字节数, 文件长度, CRC32（大端）
FEC_TABLE = {'ldpc12': (3, 6), 'ldpc34': (3, 12), 'none': None}   # 规则 (dv, dc)，码率 ≈ 1 − dv/dc


# ---------------- 封装 ----------------
def pack_file(path):
    """文件 -> 字节包：文件头 + 文件名(UTF-8) + 内容。CRC32 覆盖内容，供接收端校验。"""
    with open(path, 'rb') as f:
        data = f.read()
    name = os.path.basename(path).encode('utf-8')[:65535]
    return HDR.pack(MAGIC, VERSION, len(name), len(data), zlib.crc32(data)) + name + data


def unpack_file(buf):
    """字节包 -> (文件名, 内容, CRC 是否通过)；文件头损坏时抛出 ValueError。"""
    if len(buf) < HDR.size:
        raise ValueError('数据过短，没有完整文件头')
    magic, ver, nlen, size, crc = HDR.unpack_from(buf)
    if magic != MAGIC or ver != VERSION:
        raise ValueError('文件头损坏（魔数/版本不符）')
    name = bytes(buf[HDR.size:HDR.size + nlen]).decode('utf-8', 'replace')
    data = bytes(buf[HDR.size + nlen:HDR.size + nlen + size])
    return name, data, len(data) == size and zlib.crc32(data) == crc


# ---------------- 信道编码与加扰 ----------------
def make_code(fec_kind='ldpc34', n=1944, seed=1):
    """按协议生成 LDPC 码（收发两端同一种子得到同一校验矩阵）；'none' 返回 None。"""
    if FEC_TABLE[fec_kind] is None:
        return None
    dv, dc = FEC_TABLE[fec_kind]
    return fec.ldpc_make(n, dv, dc, rng=seed)


def scrambler(n, seed=1):
    """加扰序列（固定种子伪随机序列，收发共享）：使比特 0/1 均衡、星座点等概率，
    避免文件中大量重复字节（如全 0）造成长连 0 与谱线。"""
    return np.random.default_rng(seed + 101).integers(0, 2, n, dtype=np.uint8)


def encode_bytes(packet, code=None, seed=1):
    """字节包 -> 已编码、已加扰的比特流。返回 (coded (n,), info (k_tot,), n_cw 码字数)。"""
    info = np.unpackbits(np.frombuffer(packet, np.uint8))
    if code is None:
        coded, n_cw = info.copy(), 0
    else:
        k = code['k']
        n_cw = int(np.ceil(info.size / k))
        u = np.zeros(n_cw * k, np.uint8)
        u[:info.size] = info                                      # 末尾补零凑满整数个码字
        coded = fec.ldpc_encode(u.reshape(n_cw, k), code).ravel()
    return coded ^ scrambler(coded.size, seed), info, n_cw


# ---------------- 发射：文件 -> tx_signal 格式 ----------------
def file_tx_signal(path, cfg, fec_kind='ldpc34', ldpc_n=1944, min_sym=2 ** 14):
    """文件 -> 编码 -> 双偏振 M-QAM -> 成帧 -> RRC 成形，返回与 system.tx_signal 相同格式的 dict。

    参数: cfg system.Config（用到 mod/M/sps/beta/train_len/pilot_period/seed）; fec_kind 'ldpc34' | 'ldpc12' | 'none';
          min_sym 每偏振最少数据符号数（小文件用伪随机比特填充，保证接收端 MIMO 均衡有足够收敛长度）
    返回 dict: const, bits (2, nd·k) 实际发送比特（含填充）, frame (2, Nf), kind (Nf,), wave (2, Nf·sps)，
              另加 codec（解码所需协议信息：fec_kind/ldpc_n/seed/n_cw/n_coded/n_info/file/size/code）
    """
    c = tx.constellation(cfg.mod, cfg.M)
    k = int(np.log2(cfg.M))
    code = make_code(fec_kind, ldpc_n, cfg.seed)
    packet = pack_file(path)
    coded, info, n_cw = encode_bytes(packet, code, cfg.seed)
    blk = cfg.pilot_period - 1
    nd = max(int(np.ceil(coded.size / (2 * k))), min_sym)
    nd = int(np.ceil(nd / blk)) * blk                             # 每偏振数据符号数，取整到整数个导频块
    rng = np.random.default_rng(cfg.seed + 202)
    bits = rng.integers(0, 2, 2 * nd * k, dtype=np.uint8)         # 填充比特
    bits[:coded.size] = coded
    bits = bits.reshape(2, -1)
    train = c[rng.integers(0, cfg.M, (2, cfg.train_len))]
    pilot = c[rng.integers(0, cfg.M, (2, nd // blk))]
    f, kind = tx.frame(tx.modulate(bits, c), train, pilot, cfg.pilot_period)
    codec = dict(fec_kind=fec_kind, ldpc_n=ldpc_n, seed=cfg.seed, n_cw=n_cw, n_coded=coded.size,
                 n_info=info.size, file=os.path.basename(path), size=os.path.getsize(path), code=code)
    return dict(const=c, bits=bits, frame=f, kind=kind, wave=normalize(tx.pulse_shape(f, cfg.sps, cfg.beta)),
                codec=codec)


# ---------------- 接收：rx_dsp 输出 -> 文件 ----------------
def decode_file(y, ref, out_dir=None, prefix='rx_', iters=50):
    """rx_dsp 的 res['y']（与 ref['frame'] 逐符号对齐的整帧）-> LLR -> 解扰 -> LDPC 译码 -> 文件。

    参数: y (2, Nf) 接收符号; ref file_tx_signal 的返回值（只用到协议信息与训练/导频，不用发送数据）;
          out_dir 给出时把还原的文件写为 out_dir/prefix+文件名
    返回 dict: name, data (bytes 或 None), crc_ok, path, ber_pre 纠错前 BER, n_fail 未通过校验的码字数, packet
    """
    c, f, kind, cd = ref['const'], ref['frame'], ref['kind'], ref['codec']
    code = cd['code'] if 'code' in cd else make_code(cd['fec_kind'], cd['ldpc_n'], cd['seed'])
    sig2 = np.mean(np.abs(y[:, kind > 0] - f[:, kind > 0]) ** 2, -1)        # 每偏振噪声方差（训练+导频）
    L = np.concatenate([tx.llr(y[p, kind == 0], c, sig2[p]) for p in range(2)])[:cd['n_coded']]
    scr = scrambler(L.size, cd['seed'])
    L = L * (1 - 2.0 * scr)                                                 # LLR 域解扰：扰码为 1 的位翻转符号
    tx_coded = ref['bits'].ravel()[:cd['n_coded']] ^ scr                    # 仅用于统计纠错前 BER
    ber_pre = float(np.mean((L < 0) != tx_coded))
    if code is None:
        info, n_fail = (L < 0).astype(np.uint8), 0
    else:
        b, ok = fec.ldpc_decode(L.reshape(cd['n_cw'], -1), code, iters=iters)
        info, n_fail = b[:, code['free']].ravel(), int(np.sum(~ok))
    packet = np.packbits(info[:cd['n_info']]).tobytes()
    out = dict(name=None, data=None, crc_ok=False, path=None, ber_pre=ber_pre, n_fail=n_fail, packet=packet)
    try:
        out['name'], out['data'], out['crc_ok'] = unpack_file(packet)
    except (ValueError, struct.error):
        return out                                                          # 文件头损坏：无法还原
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out['path'] = os.path.join(out_dir, prefix + os.path.basename(out['name']))
        with open(out['path'], 'wb') as fh:
            fh.write(out['data'])
    return out
