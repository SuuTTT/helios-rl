#!/usr/bin/env bash
# Iteration 5 live dashboard. Prints best MPPI / last eval / box status
# for every active run. Cheap, idempotent — run as often as you want.
#
# Usage:  bash scripts/iter5_dashboard.sh

set -u
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; CYN='\033[0;36m'; NC='\033[0m'

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

REPO=/root/helios-rl
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "    ITERATION 5 — HopperHop sweep  ($(date -u +%FT%TZ))"
echo "═══════════════════════════════════════════════════════════════════════════════"

# Local 4070 Ti
echo -e "${CYN}── Local 4070 Ti ──${NC}"
for s in 1 2 3; do
    f="$REPO/exp/tdmpc_glass/HopperHop_phasev/seed_${s}.csv"
    if [[ -f $f ]]; then
        printf "  Phase-v seed %d (Path 7):  best %-30s   last %s\n" $s "$(best_mppi $f)" "$(last_eval $f)"
    fi
done

# Remote 3060Ti
echo -e "${CYN}── ssh3:11271 (3060Ti) ──${NC}"
ssh -p 11271 -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@ssh3.vast.ai \
  "for s in 1 2 3; do
     f=/root/helios-rl/exp/tdmpc_glass/HopperHop_phasey_3060ti/seed_\$s.csv
     if [[ -f \$f ]]; then
       best=\$(awk -F, 'NR>1 && \$3==\"mppi\" {if(\$2+0>m){m=\$2+0; ms=\$1}} END{if(m>0) printf \"%.1f @ %s\", m, ms; else printf \"—\"}' \$f)
       last=\$(awk -F, 'NR>1 && \$3==\"mppi\"' \$f | tail -1 | awk -F, '{printf \"step=%s MPPI=%.1f\", \$1, \$2+0}')
       printf '  Phase-y seed %d (Path 10): best %-30s   last %s\n' \$s \"\$best\" \"\$last\"
     fi
   done" 2>/dev/null

# Remote 2x3060 (one host, both GPUs)
echo -e "${CYN}── 78.83.187.54:17637 (2x3060) ──${NC}"
ssh -p 17637 -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@78.83.187.54 \
  "for s in 2 3; do
     f=/root/helios-rl/exp/tdmpc_glass/HopperHop_phasev_2x3060/seed_\$s.csv
     if [[ -f \$f ]]; then
       best=\$(awk -F, 'NR>1 && \$3==\"mppi\" {if(\$2+0>m){m=\$2+0; ms=\$1}} END{if(m>0) printf \"%.1f @ %s\", m, ms; else printf \"—\"}' \$f)
       last=\$(awk -F, 'NR>1 && \$3==\"mppi\"' \$f | tail -1 | awk -F, '{printf \"step=%s MPPI=%.1f\", \$1, \$2+0}')
       printf '  GPU0 Phase-v seed %d (Path 7):  best %-30s   last %s\n' \$s \"\$best\" \"\$last\"
     fi
   done
   for s in 1 2; do
     f=/root/helios-rl/exp/tdmpc_glass/HopperHop_phasex_2x3060/seed_\$s.csv
     if [[ -f \$f ]]; then
       best=\$(awk -F, 'NR>1 && \$3==\"mppi\" {if(\$2+0>m){m=\$2+0; ms=\$1}} END{if(m>0) printf \"%.1f @ %s\", m, ms; else printf \"—\"}' \$f)
       last=\$(awk -F, 'NR>1 && \$3==\"mppi\"' \$f | tail -1 | awk -F, '{printf \"step=%s MPPI=%.1f\", \$1, \$2+0}')
       printf '  GPU1 Phase-x seed %d (Path 9):  best %-30s   last %s\n' \$s \"\$best\" \"\$last\"
     fi
   done" 2>/dev/null

# Remote 4060
echo -e "${CYN}── ssh6:11115 (4060) ──${NC}"
ssh -p 11115 -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@ssh6.vast.ai \
  "for f in /root/helios-rl/exp/tdmpc_glass/HopperHop_phasep_remote_3m/seed_6.csv \
            /root/helios-rl/exp/tdmpc_glass/HopperHop_phasev_4060/seed_3.csv; do
     [[ -f \$f ]] || continue
     name=\$(basename \$(dirname \$f) | sed 's/HopperHop_//; s/_remote_3m//; s/_4060//')
     seed=\$(basename \$f .csv | sed 's/seed_//')
     best=\$(awk -F, 'NR>1 && \$3==\"mppi\" {if(\$2+0>m){m=\$2+0; ms=\$1}} END{if(m>0) printf \"%.1f @ %s\", m, ms; else printf \"—\"}' \$f)
     last=\$(awk -F, 'NR>1 && \$3==\"mppi\"' \$f | tail -1 | awk -F, '{printf \"step=%s MPPI=%.1f\", \$1, \$2+0}')
     printf '  %s seed %s: best %-30s   last %s\n' \"\$name\" \"\$seed\" \"\$best\" \"\$last\"
   done" 2>/dev/null

echo
echo -e "${YLW}Reference: Phase-p winner s4 hit MPPI=52.7 @ 1.25M, then surged to 538.${NC}"
echo -e "${YLW}Target: MPPI > 500 (ideally > 700) on ≥3 of 5 seeds.${NC}"
