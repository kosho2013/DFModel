
�
7
Add_Prev_Layer(0J�` �N5 `�LM `�L`��`��N�
6
LayerNorm_1(0J�` �N5 `�LM `�L`��`��N�
8
Q(0:+`��` �N5 `�LE  MM `�L`����`��N�
8
K(0:+`��` �N5 `�LE  MM `�L`����`��N�
8
V(0:+`��` �N5 `�LE  MM `�L`����`��N�
B

MHA_GEMM_1(0B,`�N� �N5 `�LE `�LMP`�������N�
3
SOFTMAX(0J `�N �N5PMP`�����N�
5
	DropOut_1(0J `�N �N5PMP`�����N�
A

MHA_GEMM_2	(0B+`��N �N5 `�LEPM `�L`����N��N�
G
	PROJ_GEMM
(0:2�`�` �N5 `�LE  MM `�L`hu `�L��`����N�
4
	DropOut_2(0J�` �N5 `�LM `�L`��`��N�
6
Add_1	(0R%�` �N5 `�LM `�L`��`��N�� `�L
6
LayerNorm_2
(0J�` �N5 `�LM `�L`��`��N�
<
FFN0(0:,���` �N5 `�LE  NM `�M`�� ��`��N�
0
GeLU(0J �� �N5 `�MM `�M`�� ��N�
C
FFN1(0:3�`�� �N5 `�ME  NM `�L`hu `�L��`�� ��N�
4
	DropOut_3(0J�` �N5 `�LM `�L`��`��N�
6
Add_2(0R%�` �N5 `�LM `�L`��`��N�� `�L7-   �0= `�LE `�LPZAdd_Prev_LayerbLayerNorm_1�%0= `�LE `�LPZLayerNorm_1bQ�%0= `�LE `�LPZLayerNorm_1bK�%0= `�LE `�LPZLayerNorm_1bV�$0= `�LE @KPZQb
MHA_GEMM_1�$0= `�LE @KPZKb
MHA_GEMM_1�*0=PE �>NPZ
MHA_GEMM_1bSOFTMAX�)0=PE �>NPZSOFTMAXb	DropOut_1�$	0= `�LE @KP	ZVb
MHA_GEMM_2�,	0=PE �>NP
Z	DropOut_1b
MHA_GEMM_2�,	
0= `�LE @KPZ
MHA_GEMM_2b	PROJ_GEMM�0
-   �0= `�LE `�LPZ	PROJ_GEMMb	DropOut_2�,-   �0= `�LE `�LPZ	DropOut_2bAdd_1�1-   �0
= `�LE `�LPZAdd_Prev_LayerbAdd_1�.-   �0= `�LE `�LPZAdd_1bLayerNorm_2�--   �0= `�LE `�LPZLayerNorm_2bFFN0�&-   �0= `�ME @LPZFFN0bGeLU�&-   �0= `�ME @LPZGeLUbFFN1�+-   �0= `�LE `�LPZFFN1b	DropOut_3�,-   �0= `�LE `�LPZ	DropOut_3bAdd_2�(-   �0= `�LE `�LPZAdd_1bAdd_2�8� -  �QU  �?�E  �D"��  �C�TP�
  zE   Q(   @  �A  �?% � G-�d*<E��T=M(a&>U �;D"+:�`�` �N(`X`h�Nxpx��  �?��*o�:�