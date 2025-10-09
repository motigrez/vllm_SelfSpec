def cal_avg_accept_length(accept_rate, draft_k):
    avg_accept_length = 0
    for i in range(1, draft_k+1):
        avg_accept_length += i * (accept_rate ** (i - 1)) * (1 - accept_rate)
    avg_accept_length += (draft_k+1) * (accept_rate ** draft_k)
    return avg_accept_length
        
for accept_rate in [0.5, 0.6, 0.7, 0.8, 0.9]:
    for draft_k in [8]:
        print(f"accept_rate: {accept_rate}, draft_k: {draft_k}, avg_accept_length: {cal_avg_accept_length(accept_rate, draft_k)}")
