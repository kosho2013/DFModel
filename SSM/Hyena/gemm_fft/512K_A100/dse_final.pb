
£$
2
Q(0:)   5   LE   EM   L`   
4
K(0:)   5   LE   EM   L`   
4
V(0:)   5   LE   EM   L` , 
N
Q_stage0 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
Q_stage1 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
Q_stage2 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
Q_stage3 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
Q_stage4 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
K_stage0 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
K_stage1	 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
K_stage2
 (0:8   5   JM   J` ) ΐ>} ρ   Lρ   L
N
K_stage3 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
K_stage4 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
P

QKi_stage0 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
P

QKi_stage1 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
P

QKi_stage2	 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
P

QKi_stage3
 (0:8   5   JM   J` $ ΐ>} ρ   Lρ   L
P

QKi_stage4 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
R
inter_stage0 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
R
inter_stage1 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
R
inter_stage2 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
R
inter_stage3 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
R
inter_stage4 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
V_stage0 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
V_stage1 (0:8   5   JM   J`  ΐ>} ρ   Lρ   L
N
V_stage2 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
V_stage3 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
N
V_stage4 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
T
interVi_stage0 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
T
interVi_stage1 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
T
interVi_stage2 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
T
interVi_stage3 (0:8   5   JM   J`   ΐ>} ρ   Lρ   L
T
interVi_stage4  (0:8   5   JM   J`  ΐ>} ρ   Lρ   L
@

QKmultiply! (0R(@ 5   JM   J`@ υ   J} 
=
softmax" (0R(@ 5   JM   J`@ υ   J} 
D
interVmultiply# (0R(@ 5   JM   J`@ υ   J} 
=
FFN0$ (0:+   5   LE   FM   M`  
=
FFN1% (0:+   5   ME   FM   L`  )0=   JE   JPZQ_stage0bQ_stage1 )0=   JE   JPZQ_stage1bQ_stage2 )0=   JE   JPZQ_stage2bQ_stage3 )0=   JE   JPZQ_stage3bQ_stage4 )	0=   JE   JPZK_stage0bK_stage1 )	
0=   JE   JP	ZK_stage1bK_stage2 )
0=   JE   JP
ZK_stage2bK_stage3 )0=   JE   JPZK_stage3bK_stage4 -0=   JE   JPZ
QKi_stage0b
QKi_stage1 -0=   JE   JPZ
QKi_stage1b
QKi_stage2 -0=   JE   JPZ
QKi_stage2b
QKi_stage3 -0=   JE   JPZ
QKi_stage3b
QKi_stage4 10=   JE   JPZinter_stage0binter_stage1 10=   JE   JPZinter_stage1binter_stage2 10=   JE   JPZinter_stage2binter_stage3 10=   JE   JPZinter_stage3binter_stage4 )0=   JE   JPZV_stage0bV_stage1 )0=   JE   JPZV_stage1bV_stage2 )0=   JE   JPZV_stage2bV_stage3 )0=   JE   JPZV_stage3bV_stage4 50=   JE   JPZinterVi_stage0binterVi_stage1 50=   JE   JPZinterVi_stage1binterVi_stage2 50=   JE   JPZinterVi_stage2binterVi_stage3 5 0=   JE   JPZinterVi_stage3binterVi_stage4 +!0=   JE   JP2ZQ_stage4b
QKmultiply +!0=   JE   JP3ZK_stage4b
QKmultiply -!0=   JE  JP4Z
QKmultiplyb
QKi_stage0 *"0=   JE   JP5Z
QKi_stage4bsoftmax ,"0=   JE   JP6Zsoftmaxbinter_stage0 3#0=   JE   JP7Zinter_stage4binterVmultiply /#0=   JE   JP8ZV_stage4binterVmultiply 5#0=   JE   JP9ZinterVmultiplybinterVi_stage0 + $0=   JE   JP:ZinterVi_stage4bFFN0 !$%0=   ME @KP;ZFFN0bFFN1  0=   LE  JP<ZQbQ_stage0 "0=   LE  JP=ZKbK_stage0 "0=   LE  JP>ZVbV_stage0 -° Uαz΄?" ­   A²DP’
   F  ΐQ-   @  ΐA  ?%F-τύT=5τύT=EτύT=M(a&>UeZήC"%:   (X`h xpxΐ>*o:΄