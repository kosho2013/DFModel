
£$
2
Q(0:)  5  KE   EM  K`  Ν 
4
K(0:)  5  KE   EM  K`  Ν 
4
V(0:)  5  KE   EM  K`  Ν 
N
Q_stage0 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
Q_stage1 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
Q_stage2 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
Q_stage3 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
Q_stage4 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
K_stage0 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
K_stage1	 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
K_stage2
 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
K_stage3 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
K_stage4 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
P

QKi_stage0 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
P

QKi_stage1 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
P

QKi_stage2	 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
P

QKi_stage3
 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
P

QKi_stage4 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
R
inter_stage0 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
R
inter_stage1 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
R
inter_stage2 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
R
inter_stage3 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
R
inter_stage4 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
V_stage0 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
V_stage1 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
V_stage2 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
V_stage3 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
N
V_stage4 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
T
interVi_stage0 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
T
interVi_stage1 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
T
interVi_stage2 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
T
interVi_stage3 (0:8   5  IM  I`   ΐ>} ρ  Kρ  K
T
interVi_stage4  (0:8   5  IM  I` 9 ΐ>} ρ  Kρ  K
@

QKmultiply! (0R(  5  IM  I`  υ  I} 
=
softmax" (0R(  5  IM  I`  υ  I} 
D
interVmultiply# (0R(  5  IM  I`  υ  I} 
=
FFN0$ (0:+  5  KE   FM  L` Ν 
=
FFN1% (0:+  5  LE   FM  K` Ν )0=  IE  IPZQ_stage0bQ_stage1 )0=  IE  IPZQ_stage1bQ_stage2 )0=  IE  IPZQ_stage2bQ_stage3 )0=  IE  IPZQ_stage3bQ_stage4 )	0=  IE  IPZK_stage0bK_stage1 )	
0=  IE  IP	ZK_stage1bK_stage2 )
0=  IE  IP
ZK_stage2bK_stage3 )0=  IE  IPZK_stage3bK_stage4 -0=  IE  IPZ
QKi_stage0b
QKi_stage1 -0=  IE  IPZ
QKi_stage1b
QKi_stage2 -0=  IE  IPZ
QKi_stage2b
QKi_stage3 -0=  IE  IPZ
QKi_stage3b
QKi_stage4 10=  IE  IPZinter_stage0binter_stage1 10=  IE  IPZinter_stage1binter_stage2 10=  IE  IPZinter_stage2binter_stage3 10=  IE  IPZinter_stage3binter_stage4 )0=  IE  IPZV_stage0bV_stage1 )0=  IE  IPZV_stage1bV_stage2 )0=  IE  IPZV_stage2bV_stage3 )0=  IE  IPZV_stage3bV_stage4 50=  IE  IPZinterVi_stage0binterVi_stage1 50=  IE  IPZinterVi_stage1binterVi_stage2 50=  IE  IPZinterVi_stage2binterVi_stage3 5 0=  IE  IPZinterVi_stage3binterVi_stage4 +!0=  IE  IP2ZQ_stage4b
QKmultiply +!0=  IE  IP3ZK_stage4b
QKmultiply -!0=  IE  IP4Z
QKmultiplyb
QKi_stage0 *"0=  IE  IP5Z
QKi_stage4bsoftmax ,"0=  IE  IP6Zsoftmaxbinter_stage0 3#0=  IE  IP7Zinter_stage4binterVmultiply /#0=  IE  IP8ZV_stage4binterVmultiply 5#0=  IE  IP9ZinterVmultiplybinterVi_stage0 + $0=  IE  IP:ZinterVi_stage4bFFN0 !$%0=  LE ΝLKP;ZFFN0bFFN1  0=  KE ΝLJP<ZQbQ_stage0 "0=  KE ΝLJP=ZKbK_stage0 "0=  KE ΝLJP>ZVbV_stage0 2  -   OUΝΜΜ?" ­   A²DP’
   F  ΐQ-   @  ΐA  ?%F-τύT=5τύT=EτύT=M(a&>UeZήC"%:  (X`hxpxΐ>*o:΄