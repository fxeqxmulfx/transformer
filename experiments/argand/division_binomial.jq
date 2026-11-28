# Upper tails used by the registered seed cohorts. Both null boundaries
# are dyadic and the integer coefficients are exact at these sample sizes.
def choose($n;$k): reduce range(1;$k+1) as $i (1; . * ($n-$k+$i) / $i);
def binomial_upper_tail($n;$k;$p):
  [range($k;$n+1) as $i | choose($n;$i)*pow($p;$i)*pow(1-$p;$n-$i)] | add;
