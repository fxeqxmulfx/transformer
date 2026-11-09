# Recheck serialized per-query endpoints independently of the Rust counters.
def tol: map(fabs) | max | (1+.)*1e-8;
def encloses($lo;$hi;$x):
  ([$lo,$hi,$x]|tol) as $t | $lo-$t<=$x and $x<=$hi+$t;
(.rows|length==9409) and
([.rows[][0]]|sort==[range(9409)]) and
(.methods==["legacy","gram","local-rms","combined","directional","directional-raw"]) and
all(.rows[];
 . as $r | (length==8) and (.[7]|length==6) and
 all(.[0:7][];type=="number") and
 .[1]>=0 and .[1]<97 and .[2]>=0 and .[2]<97 and
 all(.[7][];
   . as $b |
   ([$r[3],$b[0],$b[1]]|tol) as $mt |
   ([$r[5],$b[2],$b[3]]|tol) as $lt |
   (length==4) and all(.[];type=="number") and
   $b[0]<=$b[1] and $b[2]<=$b[3] and
   encloses($b[0];$b[1];$r[4]) and encloses($b[2];$b[3];$r[6]) and
   (if $b[0]>$mt then $r[1]==$r[2] else true end) and
   (if $b[0]>$r[3]+$mt then $r[4]>$r[3] else true end) and
   (if $b[1]<$r[3]-$mt then $r[4]<$r[3] else true end) and
   (if $b[3]<$r[5]-$lt then $r[6]<$r[5] else true end) and
   (if $b[2]>$r[5]+$lt then $r[6]>$r[5] else true end)))
