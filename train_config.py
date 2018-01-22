# Audio
num_mels = 80
num_freq = 1024
sample_rate = 20000
frame_length_ms = 50.
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
hidden_size = 128
embedding_size = 256

max_iters = 200
griffin_lim_iters = 60
power = 1.5
dec_out_per_step = 5
teacher_forcing_ratio = 1.0

epochs = 10000
lr = 0.001
decay_step = [500000, 1000000, 2000000]
log_step = 100
save_step = 2000

cleaners = 'english_cleaners'

data_path = '/data/shared/KeithIto/LJSpeech-1.0/'
output_path = './result'
checkpoint_path = './model_new'
