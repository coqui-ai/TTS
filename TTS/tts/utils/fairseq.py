import torch


def rehash_fairseq_vits_checkpoint(checkpoint_file):
    chk = torch.load(checkpoint_file, map_location=torch.device("cpu"))["model"]
    new_chk = {}
    for k, v in chk.items():
        if "enc_p." in k:
            new_chk[k.replace("enc_p.", "text_encoder.")] = v
        elif "dec." in k:
            new_chk[k.replace("dec.", "waveform_decoder.")] = v
        elif "enc_q." in k:
            new_chk[k.replace("enc_q.", "posterior_encoder.")] = v
        elif "flow.flows.2." in k:
            new_chk[k.replace("flow.flows.2.", "flow.flows.1.")] = v
        elif "flow.flows.4." in k:
            new_chk[k.replace("flow.flows.4.", "flow.flows.2.")] = v
        elif "flow.flows.6." in k:
            new_chk[k.replace("flow.flows.6.", "flow.flows.3.")] = v
        elif "dp.flows.0.m" in k:
            new_chk[k.replace("dp.flows.0.m", "duration_predictor.flows.0.translation")] = v
        elif "dp.flows.0.logs" in k:
            new_chk[k.replace("dp.flows.0.logs", "duration_predictor.flows.0.log_scale")] = v
        elif "dp.flows.1" in k:
            new_chk[k.replace("dp.flows.1", "duration_predictor.flows.1")] = v
        elif "dp.flows.3" in k:
            new_chk[k.replace("dp.flows.3", "duration_predictor.flows.2")] = v
        elif "dp.flows.5" in k:
            new_chk[k.replace("dp.flows.5", "duration_predictor.flows.3")] = v
        elif "dp.flows.7" in k:
            new_chk[k.replace("dp.flows.7", "duration_predictor.flows.4")] = v
        elif "dp.post_flows.0.m" in k:
            new_chk[k.replace("dp.post_flows.0.m", "duration_predictor.post_flows.0.translation")] = v
        elif "dp.post_flows.0.logs" in k:
            new_chk[k.replace("dp.post_flows.0.logs", "duration_predictor.post_flows.0.log_scale")] = v
        elif "dp.post_flows.1" in k:
            new_chk[k.replace("dp.post_flows.1", "duration_predictor.post_flows.1")] = v
        elif "dp.post_flows.3" in k:
            new_chk[k.replace("dp.post_flows.3", "duration_predictor.post_flows.2")] = v
        elif "dp.post_flows.5" in k:
            new_chk[k.replace("dp.post_flows.5", "duration_predictor.post_flows.3")] = v
        elif "dp.post_flows.7" in k:
            new_chk[k.replace("dp.post_flows.7", "duration_predictor.post_flows.4")] = v
        elif "dp." in k:
            new_chk[k.replace("dp.", "duration_predictor.")] = v
        else:
            new_chk[k] = v
    return new_chk
