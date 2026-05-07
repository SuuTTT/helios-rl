with open("/workspace/cleanrl/cleanrl/ppo.py", "r") as f:
    lines = f.readlines()
with open("/workspace/cleanrl/cleanrl/ppo.py", "w") as f:
    skip = False
    for line in lines:
        if 'if "final_info" in infos:' in line:
            f.write('            if "episode" in infos:\n')
            f.write('                for i, done in enumerate(next_done):\n')
            f.write('                    if done and infos["_episode"][i]:\n')
            f.write('                        print(f"global_step={global_step}, episodic_return={infos[\'episode\'][\'r\'][i]}")\n')
            f.write('                        writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)\n')
            f.write('                        writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)\n')
            skip = True
        elif skip and 'writer.add_scalar("charts/episodic_length"' in line:
            skip = False
            continue
        elif not skip:
            f.write(line)
