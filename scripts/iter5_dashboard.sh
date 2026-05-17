#!/usr/bin/env bash
# Iteration 5 live dashboard. Shows for each box:
#   - best MPPI / last eval per run
#   - which python process is actually running (PID, etime, seed, NS)
#   - GPU + CPU utilization
# Designed for fast iteration awareness — run anytime.
#
# Usage:  bash scripts/iter5_dashboard.sh
set -u
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; CYN='\033[0;36m'; BLD='\033[1m'; NC='\033[0m'

best_mppi() {
    local csv=$1
    [[ -f $csv ]] || { echo "—"; return; }
    awk -F, 'NR>1 && $3=="mppi" {if($2+0>m){m=$2+0; ms=$1}} END{if(m>0) printf "%.1f @ %s", m, ms; else printf "—"}' "$csv"
}
last_eval() {
    local csv=$1
    [[ -f $csv ]] || { echo "—"; return; }
    awk -F, 'NR>1 && $3=="mppi"' "$csv" | tail -1 | awk -F, '{printf "step=%s MPPI=%.1f", $1, $2+0}'
}

# Highlight any best >= 500 in green
fmt_best() {
    local raw="$1"
    if [[ "$raw" =~ ^([0-9]+)\.[0-9]+ ]] && (( ${BASH_REMATCH[1]} >= 500 )); then
        echo -e "${GRN}${BLD}${raw}${NC}"
    else
        echo "$raw"
    fi
}

REPO=/root/helios-rl
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "    ITERATION 5 — HopperHop sweep  ($(date -u +%FT%TZ))"
echo "═══════════════════════════════════════════════════════════════════════════════"

# ───────────────────────────────────────────────── Local 4070 Ti
echo -e "${CYN}── Local 4070 Ti (12GB) ──${NC}"
ps -eo pid,etime,cmd --no-headers 2>/dev/null | grep -E "run_benchmark" | grep -v grep | \
  awk '{ pid=$1; et=$2;
         for(i=3;i<=NF;i++){
             if($i=="--seed") s=$(i+1);
             if($i=="--mppi_n_samples") ns=$(i+1);
             if($i=="--use_cluster_obs") tag="Path7";
         }
         if(ns=="2048") tag="Path9(NS=2048)";
         else if(ns=="1024") tag="Path9-NS1024";
         else if(!tag) tag="Path-other";
         printf "  ▶ PID=%s  etime=%s  seed=%s  %s\n", pid, et, s, tag
       }'
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)
cpu=$(top -bn1 2>/dev/null | grep -E "^%Cpu" | awk '{printf "%.0f%%", 100-$8}')
echo "  GPU: $gpu | CPU: $cpu"
for s in 1 2 3; do
    f="$REPO/exp/tdmpc_glass/HopperHop_phasev/seed_${s}.csv"
    [[ -f $f ]] && printf "  Phase-v s%d (Path 7):     best %-30b   last %s\n" $s "$(fmt_best "$(best_mppi $f)")" "$(last_eval $f)"
done
for s in 1 2 3 4 5; do
    f="$REPO/exp/tdmpc_glass/HopperHop_phasex_local/seed_${s}.csv"
    [[ -f $f ]] && printf "  Phase-x s%d (Path 9):     best %-30b   last %s\n" $s "$(fmt_best "$(best_mppi $f)")" "$(last_eval $f)"
done

# ───────────────────────────────────────────────── Remote 3060Ti
echo -e "${CYN}── ssh3:11271 (3060Ti, 8GB) ──${NC}"
ssh -p 11271 -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@ssh3.vast.ai \
  "ps -eo pid,etime,cmd --no-headers | grep -E 'run_benchmark' | grep -v grep | \
     awk '{pid=\$1; et=\$2; for(i=3;i<=NF;i++){if(\$i==\"--seed\")s=\$(i+1);if(\$i==\"--glass_num_super_clusters\")tag=\"Path10\";} printf \"  ▶ PID=%s etime=%s seed=%s %s\n\", pid, et, s, (tag?tag:\"Phase-y\")}'
   echo \"  GPU: \$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | head -1) | CPU: \$(top -bn1 | grep '^%Cpu' | awk '{printf \"%.0f%%\", 100-\$8}')\"
   for s in 1 2 3; do
     f=/root/helios-rl/exp/tdmpc_glass/HopperHop_phasey_3060ti/seed_\$s.csv
     [[ -f \$f ]] && {
       best=\$(awk -F, 'NR>1 && \$3==\"mppi\" {if(\$2+0>m){m=\$2+0; ms=\$1}} END{if(m>0) printf \"%.1f @ %s\", m, ms; else printf \"—\"}' \$f)
       last=\$(awk -F, 'NR>1 && \$3==\"mppi\"' \$f | tail -1 | awk -F, '{printf \"step=%s MPPI=%.1f\", \$1, \$2+0}')
       printf '  Phase-y s%s (Path 10):    best %-30s   last %s\n' \$s \"\$best\" \"\$last\"
     }
   done" 2>/dev/null

# ───────────────────────────────────────────────── Remote 2x3060
echo -e "${CYN}── 78.83.187.54:17637 (2× 3060 Laptop 6GB each) ──${NC}"
ssh -p 17637 -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@78.83.187.54 \
  "ps -eo pid,etime,cmd --no-headers | grep -E 'run_benchmark' | grep -v grep | \
     awk '{pid=\$1; et=\$2; for(i=3;i<=NF;i++){if(\$i==\"--seed\")s=\$(i+1);if(\$i==\"--mppi_n_samples\")ns=\$(i+1);} printf \"  ▶ PID=%s etime=%s seed=%s NS=%s\n\", pid, et, s, (ns?ns:\"512\")}'
   nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -2 | awk -F', ' '{printf \"  GPU%s: %s, %s / %s\n\", \$1, \$2, \$3, \$4}'
   echo \"  CPU: \$(top -bn1 | grep '^%Cpu' | awk '{printf \"%.0f%%\", 100-\$8}')\"
   for s in 1 2; do
     f=/root/helios-rl/exp/tdmpc_glass/HopperHop_phasex_2x3060/seed_\$s.csv
     [[ -f \$f ]] && {
       best=\$(awk -F, 'NR>1 && \$3==\"mppi\" {if(\$2+0>m){m=\$2+0; ms=\$1}} END{if(m>0) printf \"%.1f @ %s\", m, ms; else printf \"—\"}' \$f)
       last=\$(awk -F, 'NR>1 && \$3==\"mppi\"' \$f | tail -1 | awk -F, '{printf \"step=%s MPPI=%.1f\", \$1, \$2+0}')
       printf '  Phase-x s%s (Path 9):     best %-30s   last %s\n' \$s \"\$best\" \"\$last\"
     }
   done
   for s in 5; do
     f=/root/helios-rl/exp/tdmpc_glass/HopperHop_phasex_ns1024/seed_\$s.csv
     [[ -f \$f ]] && {
       best=\$(awk -F, 'NR>1 && \$3==\"mppi\" {if(\$2+0>m){m=\$2+0; ms=\$1}} END{if(m>0) printf \"%.1f @ %s\", m, ms; else printf \"—\"}' \$f)
       last=\$(awk -F, 'NR>1 && \$3==\"mppi\"' \$f | tail -1 | awk -F, '{printf \"step=%s MPPI=%.1f\", \$1, \$2+0}')
       printf '  Phase-x s%s (NS=1024):    best %-30s   last %s\n' \$s \"\$best\" \"\$last\"
     }
   done" 2>/dev/null

# ───────────────────────────────────────────────── Remote 4060
echo -e "${CYN}── ssh6:11115 (4060, 8GB) ──${NC}"
ssh -p 11115 -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@ssh6.vast.ai \
  "ps -eo pid,etime,cmd --no-headers | grep -E 'run_benchmark' | grep -v grep | \
     awk '{pid=\$1; et=\$2; for(i=3;i<=NF;i++){if(\$i==\"--seed\")s=\$(i+1);if(\$i==\"--use_cluster_obs\")tag=\"Path7\";} printf \"  ▶ PID=%s etime=%s seed=%s %s\n\", pid, et, s, (tag?tag:\"Phase-p\")}'
   echo \"  GPU: \$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | head -1) | CPU: \$(top -bn1 | grep '^%Cpu' | awk '{printf \"%.0f%%\", 100-\$8}')\"
   for f in /root/helios-rl/exp/tdmpc_glass/HopperHop_phasep_remote_3m/seed_6.csv \
            /root/helios-rl/exp/tdmpc_glass/HopperHop_phasev_4060/seed_3.csv; do
     [[ -f \$f ]] || continue
     name=\$(basename \$(dirname \$f) | sed 's/HopperHop_//; s/_remote_3m//; s/_4060//')
     seed=\$(basename \$f .csv | sed 's/seed_//')
     best=\$(awk -F, 'NR>1 && \$3==\"mppi\" {if(\$2+0>m){m=\$2+0; ms=\$1}} END{if(m>0) printf \"%.1f @ %s\", m, ms; else printf \"—\"}' \$f)
     last=\$(awk -F, 'NR>1 && \$3==\"mppi\"' \$f | tail -1 | awk -F, '{printf \"step=%s MPPI=%.1f\", \$1, \$2+0}')
     printf '  %s s%s:              best %-30s   last %s\n' \"\$name\" \"\$seed\" \"\$best\" \"\$last\"
   done" 2>/dev/null

echo
echo -e "${YLW}Ref: Phase-p winner s4 reached MPPI=52.7 @ 1.25M → surge → 538 final.${NC}"
echo -e "${YLW}Target: MPPI > 500 (ideally > 700) on ≥3 of 5 seeds.${NC}"
echo -e "${GRN}${BLD}Bold-green best = exceeds 500 target.${NC}"
