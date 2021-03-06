??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.12v2.6.0-101-g3aa40c3ce9d8??
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: 0*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:0*
dtype0
?
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_2/gamma
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:0*
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:0*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:0*
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_2/moving_variance
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:0*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:0@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
w
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_namedense/kernel
p
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*!
_output_shapes
:???*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
training/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *#
shared_nametraining/Adam/iter
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
_output_shapes
: *
dtype0	
|
training/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_1
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
_output_shapes
: *
dtype0
|
training/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_2
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
_output_shapes
: *
dtype0
z
training/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nametraining/Adam/decay
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
_output_shapes
: *
dtype0
?
training/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining/Adam/learning_rate
?
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
training/Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nametraining/Adam/conv2d/kernel/m
?
1training/Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
?
training/Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nametraining/Adam/conv2d/bias/m
?
/training/Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
)training/Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)training/Adam/batch_normalization/gamma/m
?
=training/Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp)training/Adam/batch_normalization/gamma/m*
_output_shapes
:*
dtype0
?
(training/Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(training/Adam/batch_normalization/beta/m
?
<training/Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOp(training/Adam/batch_normalization/beta/m*
_output_shapes
:*
dtype0
?
training/Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!training/Adam/conv2d_1/kernel/m
?
3training/Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/m*&
_output_shapes
: *
dtype0
?
training/Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining/Adam/conv2d_1/bias/m
?
1training/Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/m*
_output_shapes
: *
dtype0
?
+training/Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+training/Adam/batch_normalization_1/gamma/m
?
?training/Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_1/gamma/m*
_output_shapes
: *
dtype0
?
*training/Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*training/Adam/batch_normalization_1/beta/m
?
>training/Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_1/beta/m*
_output_shapes
: *
dtype0
?
training/Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*0
shared_name!training/Adam/conv2d_2/kernel/m
?
3training/Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/m*&
_output_shapes
: 0*
dtype0
?
training/Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_nametraining/Adam/conv2d_2/bias/m
?
1training/Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/m*
_output_shapes
:0*
dtype0
?
+training/Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*<
shared_name-+training/Adam/batch_normalization_2/gamma/m
?
?training/Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_2/gamma/m*
_output_shapes
:0*
dtype0
?
*training/Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*;
shared_name,*training/Adam/batch_normalization_2/beta/m
?
>training/Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_2/beta/m*
_output_shapes
:0*
dtype0
?
training/Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*0
shared_name!training/Adam/conv2d_3/kernel/m
?
3training/Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_3/kernel/m*&
_output_shapes
:0@*
dtype0
?
training/Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nametraining/Adam/conv2d_3/bias/m
?
1training/Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_3/bias/m*
_output_shapes
:@*
dtype0
?
+training/Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+training/Adam/batch_normalization_3/gamma/m
?
?training/Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_3/gamma/m*
_output_shapes
:@*
dtype0
?
*training/Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*training/Adam/batch_normalization_3/beta/m
?
>training/Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_3/beta/m*
_output_shapes
:@*
dtype0
?
training/Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*0
shared_name!training/Adam/conv2d_4/kernel/m
?
3training/Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_4/kernel/m*&
_output_shapes
:@@*
dtype0
?
training/Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nametraining/Adam/conv2d_4/bias/m
?
1training/Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_4/bias/m*
_output_shapes
:@*
dtype0
?
+training/Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+training/Adam/batch_normalization_4/gamma/m
?
?training/Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_4/gamma/m*
_output_shapes
:@*
dtype0
?
*training/Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*training/Adam/batch_normalization_4/beta/m
?
>training/Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_4/beta/m*
_output_shapes
:@*
dtype0
?
training/Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*-
shared_nametraining/Adam/dense/kernel/m
?
0training/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/m*!
_output_shapes
:???*
dtype0
?
training/Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nametraining/Adam/dense/bias/m
?
.training/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/m*
_output_shapes	
:?*
dtype0
?
training/Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*/
shared_name training/Adam/dense_1/kernel/m
?
2training/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/m*
_output_shapes
:	?@*
dtype0
?
training/Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nametraining/Adam/dense_1/bias/m
?
0training/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/m*
_output_shapes
:@*
dtype0
?
training/Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name training/Adam/dense_2/kernel/m
?
2training/Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/kernel/m*
_output_shapes

:@*
dtype0
?
training/Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/dense_2/bias/m
?
0training/Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
training/Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nametraining/Adam/conv2d/kernel/v
?
1training/Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
?
training/Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nametraining/Adam/conv2d/bias/v
?
/training/Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
)training/Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)training/Adam/batch_normalization/gamma/v
?
=training/Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp)training/Adam/batch_normalization/gamma/v*
_output_shapes
:*
dtype0
?
(training/Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(training/Adam/batch_normalization/beta/v
?
<training/Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOp(training/Adam/batch_normalization/beta/v*
_output_shapes
:*
dtype0
?
training/Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!training/Adam/conv2d_1/kernel/v
?
3training/Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/v*&
_output_shapes
: *
dtype0
?
training/Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining/Adam/conv2d_1/bias/v
?
1training/Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/v*
_output_shapes
: *
dtype0
?
+training/Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+training/Adam/batch_normalization_1/gamma/v
?
?training/Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_1/gamma/v*
_output_shapes
: *
dtype0
?
*training/Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*training/Adam/batch_normalization_1/beta/v
?
>training/Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_1/beta/v*
_output_shapes
: *
dtype0
?
training/Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*0
shared_name!training/Adam/conv2d_2/kernel/v
?
3training/Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/v*&
_output_shapes
: 0*
dtype0
?
training/Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_nametraining/Adam/conv2d_2/bias/v
?
1training/Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/v*
_output_shapes
:0*
dtype0
?
+training/Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*<
shared_name-+training/Adam/batch_normalization_2/gamma/v
?
?training/Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_2/gamma/v*
_output_shapes
:0*
dtype0
?
*training/Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*;
shared_name,*training/Adam/batch_normalization_2/beta/v
?
>training/Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_2/beta/v*
_output_shapes
:0*
dtype0
?
training/Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*0
shared_name!training/Adam/conv2d_3/kernel/v
?
3training/Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_3/kernel/v*&
_output_shapes
:0@*
dtype0
?
training/Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nametraining/Adam/conv2d_3/bias/v
?
1training/Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_3/bias/v*
_output_shapes
:@*
dtype0
?
+training/Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+training/Adam/batch_normalization_3/gamma/v
?
?training/Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_3/gamma/v*
_output_shapes
:@*
dtype0
?
*training/Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*training/Adam/batch_normalization_3/beta/v
?
>training/Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_3/beta/v*
_output_shapes
:@*
dtype0
?
training/Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*0
shared_name!training/Adam/conv2d_4/kernel/v
?
3training/Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_4/kernel/v*&
_output_shapes
:@@*
dtype0
?
training/Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nametraining/Adam/conv2d_4/bias/v
?
1training/Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_4/bias/v*
_output_shapes
:@*
dtype0
?
+training/Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+training/Adam/batch_normalization_4/gamma/v
?
?training/Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_4/gamma/v*
_output_shapes
:@*
dtype0
?
*training/Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*training/Adam/batch_normalization_4/beta/v
?
>training/Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_4/beta/v*
_output_shapes
:@*
dtype0
?
training/Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*-
shared_nametraining/Adam/dense/kernel/v
?
0training/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/v*!
_output_shapes
:???*
dtype0
?
training/Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nametraining/Adam/dense/bias/v
?
.training/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/v*
_output_shapes	
:?*
dtype0
?
training/Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*/
shared_name training/Adam/dense_1/kernel/v
?
2training/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/v*
_output_shapes
:	?@*
dtype0
?
training/Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nametraining/Adam/dense_1/bias/v
?
0training/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/v*
_output_shapes
:@*
dtype0
?
training/Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name training/Adam/dense_2/kernel/v
?
2training/Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/kernel/v*
_output_shapes

:@*
dtype0
?
training/Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/dense_2/bias/v
?
0training/Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? Bݗ
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer-11
layer_with_weights-10
layer-12
layer-13
layer_with_weights-11
layer-14
layer-15
layer_with_weights-12
layer-16
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?
axis
	gamma
 beta
!moving_mean
"moving_variance
#trainable_variables
$	variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
?
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2trainable_variables
3	variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
?
<axis
	=gamma
>beta
?moving_mean
@moving_variance
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
h

Ekernel
Fbias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
h

Tkernel
Ubias
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_trainable_variables
`	variables
aregularization_losses
b	keras_api
R
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
h

gkernel
hbias
itrainable_variables
j	variables
kregularization_losses
l	keras_api
R
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
h

qkernel
rbias
strainable_variables
t	variables
uregularization_losses
v	keras_api
R
wtrainable_variables
x	variables
yregularization_losses
z	keras_api
i

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?m? m?'m?(m?.m?/m?6m?7m?=m?>m?Em?Fm?Lm?Mm?Tm?Um?[m?\m?gm?hm?qm?rm?{m?|m?v?v?v? v?'v?(v?.v?/v?6v?7v?=v?>v?Ev?Fv?Lv?Mv?Tv?Uv?[v?\v?gv?hv?qv?rv?{v?|v?
?
0
1
2
 3
'4
(5
.6
/7
68
79
=10
>11
E12
F13
L14
M15
T16
U17
[18
\19
g20
h21
q22
r23
{24
|25
?
0
1
2
 3
!4
"5
'6
(7
.8
/9
010
111
612
713
=14
>15
?16
@17
E18
F19
L20
M21
N22
O23
T24
U25
[26
\27
]28
^29
g30
h31
q32
r33
{34
|35
 
?
?metrics
?layers
trainable_variables
?layer_metrics
	variables
?non_trainable_variables
 ?layer_regularization_losses
regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
?metrics
?layers
trainable_variables
?layer_metrics
	variables
?non_trainable_variables
 ?layer_regularization_losses
regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
!2
"3
 
?
?metrics
?layers
#trainable_variables
?layer_metrics
$	variables
?non_trainable_variables
 ?layer_regularization_losses
%regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
?metrics
?layers
)trainable_variables
?layer_metrics
*	variables
?non_trainable_variables
 ?layer_regularization_losses
+regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
02
13
 
?
?metrics
?layers
2trainable_variables
?layer_metrics
3	variables
?non_trainable_variables
 ?layer_regularization_losses
4regularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
?
?metrics
?layers
8trainable_variables
?layer_metrics
9	variables
?non_trainable_variables
 ?layer_regularization_losses
:regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
?2
@3
 
?
?metrics
?layers
Atrainable_variables
?layer_metrics
B	variables
?non_trainable_variables
 ?layer_regularization_losses
Cregularization_losses
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1

E0
F1
 
?
?metrics
?layers
Gtrainable_variables
?layer_metrics
H	variables
?non_trainable_variables
 ?layer_regularization_losses
Iregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
N2
O3
 
?
?metrics
?layers
Ptrainable_variables
?layer_metrics
Q	variables
?non_trainable_variables
 ?layer_regularization_losses
Rregularization_losses
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

T0
U1
 
?
?metrics
?layers
Vtrainable_variables
?layer_metrics
W	variables
?non_trainable_variables
 ?layer_regularization_losses
Xregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

[0
\1

[0
\1
]2
^3
 
?
?metrics
?layers
_trainable_variables
?layer_metrics
`	variables
?non_trainable_variables
 ?layer_regularization_losses
aregularization_losses
 
 
 
?
?metrics
?layers
ctrainable_variables
?layer_metrics
d	variables
?non_trainable_variables
 ?layer_regularization_losses
eregularization_losses
YW
VARIABLE_VALUEdense/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
dense/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
?
?metrics
?layers
itrainable_variables
?layer_metrics
j	variables
?non_trainable_variables
 ?layer_regularization_losses
kregularization_losses
 
 
 
?
?metrics
?layers
mtrainable_variables
?layer_metrics
n	variables
?non_trainable_variables
 ?layer_regularization_losses
oregularization_losses
[Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1

q0
r1
 
?
?metrics
?layers
strainable_variables
?layer_metrics
t	variables
?non_trainable_variables
 ?layer_regularization_losses
uregularization_losses
 
 
 
?
?metrics
?layers
wtrainable_variables
?layer_metrics
x	variables
?non_trainable_variables
 ?layer_regularization_losses
yregularization_losses
[Y
VARIABLE_VALUEdense_2/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_2/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

{0
|1
 
?
?metrics
?layers
}trainable_variables
?layer_metrics
~	variables
?non_trainable_variables
 ?layer_regularization_losses
regularization_losses
QO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

?0
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 
F
!0
"1
02
13
?4
@5
N6
O7
]8
^9
 
 
 
 
 
 
 
 
 

!0
"1
 
 
 
 
 
 
 
 
 

00
11
 
 
 
 
 
 
 
 
 

?0
@1
 
 
 
 
 
 
 
 
 

N0
O1
 
 
 
 
 
 
 
 
 

]0
^1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUEtraining/Adam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)training/Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(training/Adam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+training/Adam/batch_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*training/Adam/batch_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+training/Adam/batch_normalization_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*training/Adam/batch_normalization_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+training/Adam/batch_normalization_3/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*training/Adam/batch_normalization_3/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+training/Adam/batch_normalization_4/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*training/Adam/batch_normalization_4/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/dense/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_1/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_1/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_2/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_2/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)training/Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(training/Adam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+training/Adam/batch_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*training/Adam/batch_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+training/Adam/batch_normalization_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*training/Adam/batch_normalization_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+training/Adam/batch_normalization_3/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*training/Adam/batch_normalization_3/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+training/Adam/batch_normalization_4/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*training/Adam/batch_normalization_4/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/dense/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_1/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_1/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_2/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_2/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_4330
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?(
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1training/Adam/conv2d/kernel/m/Read/ReadVariableOp/training/Adam/conv2d/bias/m/Read/ReadVariableOp=training/Adam/batch_normalization/gamma/m/Read/ReadVariableOp<training/Adam/batch_normalization/beta/m/Read/ReadVariableOp3training/Adam/conv2d_1/kernel/m/Read/ReadVariableOp1training/Adam/conv2d_1/bias/m/Read/ReadVariableOp?training/Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp>training/Adam/batch_normalization_1/beta/m/Read/ReadVariableOp3training/Adam/conv2d_2/kernel/m/Read/ReadVariableOp1training/Adam/conv2d_2/bias/m/Read/ReadVariableOp?training/Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp>training/Adam/batch_normalization_2/beta/m/Read/ReadVariableOp3training/Adam/conv2d_3/kernel/m/Read/ReadVariableOp1training/Adam/conv2d_3/bias/m/Read/ReadVariableOp?training/Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp>training/Adam/batch_normalization_3/beta/m/Read/ReadVariableOp3training/Adam/conv2d_4/kernel/m/Read/ReadVariableOp1training/Adam/conv2d_4/bias/m/Read/ReadVariableOp?training/Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp>training/Adam/batch_normalization_4/beta/m/Read/ReadVariableOp0training/Adam/dense/kernel/m/Read/ReadVariableOp.training/Adam/dense/bias/m/Read/ReadVariableOp2training/Adam/dense_1/kernel/m/Read/ReadVariableOp0training/Adam/dense_1/bias/m/Read/ReadVariableOp2training/Adam/dense_2/kernel/m/Read/ReadVariableOp0training/Adam/dense_2/bias/m/Read/ReadVariableOp1training/Adam/conv2d/kernel/v/Read/ReadVariableOp/training/Adam/conv2d/bias/v/Read/ReadVariableOp=training/Adam/batch_normalization/gamma/v/Read/ReadVariableOp<training/Adam/batch_normalization/beta/v/Read/ReadVariableOp3training/Adam/conv2d_1/kernel/v/Read/ReadVariableOp1training/Adam/conv2d_1/bias/v/Read/ReadVariableOp?training/Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp>training/Adam/batch_normalization_1/beta/v/Read/ReadVariableOp3training/Adam/conv2d_2/kernel/v/Read/ReadVariableOp1training/Adam/conv2d_2/bias/v/Read/ReadVariableOp?training/Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp>training/Adam/batch_normalization_2/beta/v/Read/ReadVariableOp3training/Adam/conv2d_3/kernel/v/Read/ReadVariableOp1training/Adam/conv2d_3/bias/v/Read/ReadVariableOp?training/Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp>training/Adam/batch_normalization_3/beta/v/Read/ReadVariableOp3training/Adam/conv2d_4/kernel/v/Read/ReadVariableOp1training/Adam/conv2d_4/bias/v/Read/ReadVariableOp?training/Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp>training/Adam/batch_normalization_4/beta/v/Read/ReadVariableOp0training/Adam/dense/kernel/v/Read/ReadVariableOp.training/Adam/dense/bias/v/Read/ReadVariableOp2training/Adam/dense_1/kernel/v/Read/ReadVariableOp0training/Adam/dense_1/bias/v/Read/ReadVariableOp2training/Adam/dense_2/kernel/v/Read/ReadVariableOp0training/Adam/dense_2/bias/v/Read/ReadVariableOpConst*l
Tine
c2a	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference__traced_save_5751
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratetotalcounttraining/Adam/conv2d/kernel/mtraining/Adam/conv2d/bias/m)training/Adam/batch_normalization/gamma/m(training/Adam/batch_normalization/beta/mtraining/Adam/conv2d_1/kernel/mtraining/Adam/conv2d_1/bias/m+training/Adam/batch_normalization_1/gamma/m*training/Adam/batch_normalization_1/beta/mtraining/Adam/conv2d_2/kernel/mtraining/Adam/conv2d_2/bias/m+training/Adam/batch_normalization_2/gamma/m*training/Adam/batch_normalization_2/beta/mtraining/Adam/conv2d_3/kernel/mtraining/Adam/conv2d_3/bias/m+training/Adam/batch_normalization_3/gamma/m*training/Adam/batch_normalization_3/beta/mtraining/Adam/conv2d_4/kernel/mtraining/Adam/conv2d_4/bias/m+training/Adam/batch_normalization_4/gamma/m*training/Adam/batch_normalization_4/beta/mtraining/Adam/dense/kernel/mtraining/Adam/dense/bias/mtraining/Adam/dense_1/kernel/mtraining/Adam/dense_1/bias/mtraining/Adam/dense_2/kernel/mtraining/Adam/dense_2/bias/mtraining/Adam/conv2d/kernel/vtraining/Adam/conv2d/bias/v)training/Adam/batch_normalization/gamma/v(training/Adam/batch_normalization/beta/vtraining/Adam/conv2d_1/kernel/vtraining/Adam/conv2d_1/bias/v+training/Adam/batch_normalization_1/gamma/v*training/Adam/batch_normalization_1/beta/vtraining/Adam/conv2d_2/kernel/vtraining/Adam/conv2d_2/bias/v+training/Adam/batch_normalization_2/gamma/v*training/Adam/batch_normalization_2/beta/vtraining/Adam/conv2d_3/kernel/vtraining/Adam/conv2d_3/bias/v+training/Adam/batch_normalization_3/gamma/v*training/Adam/batch_normalization_3/beta/vtraining/Adam/conv2d_4/kernel/vtraining/Adam/conv2d_4/bias/v+training/Adam/batch_normalization_4/gamma/v*training/Adam/batch_normalization_4/beta/vtraining/Adam/dense/kernel/vtraining/Adam/dense/bias/vtraining/Adam/dense_1/kernel/vtraining/Adam/dense_1/bias/vtraining/Adam/dense_2/kernel/vtraining/Adam/dense_2/bias/v*k
Tind
b2`*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_restore_6046??
?
?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_3070

inputs?
%conv2d_readvariableop_conv2d_2_kernel: 02
$biasadd_readvariableop_conv2d_2_bias:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##0*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_2_bias*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????##02
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????##02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????HH : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????HH 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2560

inputs8
*readvariableop_batch_normalization_2_gamma:09
+readvariableop_1_batch_normalization_2_beta:0O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:0U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_1_layer_call_fn_4946

inputs)
batch_normalization_1_gamma: (
batch_normalization_1_beta: /
!batch_normalization_1_moving_mean: 3
%batch_normalization_1_moving_variance: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????HH 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????HH : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????HH 
 
_user_specified_nameinputs
??
?&
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4612

inputsD
*conv2d_conv2d_readvariableop_conv2d_kernel:7
)conv2d_biasadd_readvariableop_conv2d_bias:J
<batch_normalization_readvariableop_batch_normalization_gamma:K
=batch_normalization_readvariableop_1_batch_normalization_beta:a
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:g
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:H
.conv2d_1_conv2d_readvariableop_conv2d_1_kernel: ;
-conv2d_1_biasadd_readvariableop_conv2d_1_bias: N
@batch_normalization_1_readvariableop_batch_normalization_1_gamma: O
Abatch_normalization_1_readvariableop_1_batch_normalization_1_beta: e
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean: k
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance: H
.conv2d_2_conv2d_readvariableop_conv2d_2_kernel: 0;
-conv2d_2_biasadd_readvariableop_conv2d_2_bias:0N
@batch_normalization_2_readvariableop_batch_normalization_2_gamma:0O
Abatch_normalization_2_readvariableop_1_batch_normalization_2_beta:0e
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:0k
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:0H
.conv2d_3_conv2d_readvariableop_conv2d_3_kernel:0@;
-conv2d_3_biasadd_readvariableop_conv2d_3_bias:@N
@batch_normalization_3_readvariableop_batch_normalization_3_gamma:@O
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta:@e
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:@k
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:@H
.conv2d_4_conv2d_readvariableop_conv2d_4_kernel:@@;
-conv2d_4_biasadd_readvariableop_conv2d_4_bias:@N
@batch_normalization_4_readvariableop_batch_normalization_4_gamma:@O
Abatch_normalization_4_readvariableop_1_batch_normalization_4_beta:@e
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:@k
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:@=
(dense_matmul_readvariableop_dense_kernel:???6
'dense_biasadd_readvariableop_dense_bias:	??
,dense_1_matmul_readvariableop_dense_1_kernel:	?@9
+dense_1_biasadd_readvariableop_dense_1_bias:@>
,dense_2_matmul_readvariableop_dense_2_kernel:@9
+dense_2_biasadd_readvariableop_dense_2_bias:
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d/Relu?
"batch_normalization/ReadVariableOpReadVariableOp<batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp=batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH *
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????HH 2
conv2d_1/Relu?
$batch_normalization_1/ReadVariableOpReadVariableOp@batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOpAbatch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????HH : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##0*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp-conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
:0*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##02
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????##02
conv2d_2/Relu?
$batch_normalization_2/ReadVariableOpReadVariableOp@batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
:0*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOpAbatch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
:0*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????##0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_2/FusedBatchNormV3?
$batch_normalization_2/AssignNewValueAssignVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue?
&batch_normalization_2/AssignNewValue_1AssignVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
:0@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp-conv2d_3_biasadd_readvariableop_conv2d_3_bias*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!!@2
conv2d_3/Relu?
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????!!@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_3/FusedBatchNormV3?
$batch_normalization_3/AssignNewValueAssignVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue?
&batch_normalization_3/AssignNewValue_1AssignVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp-conv2d_4_biasadd_readvariableop_conv2d_4_bias*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/Relu?
$batch_normalization_4/ReadVariableOpReadVariableOp@batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOpAbatch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_4/FusedBatchNormV3?
$batch_normalization_4/AssignNewValueAssignVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue?
&batch_normalization_4/AssignNewValue_1AssignVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@?  2
flatten/Const?
flatten/ReshapeReshape*batch_normalization_4/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*!
_output_shapes
:???*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/dropout/Const?
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel*
_output_shapes
:	?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout_1/dropout/Mul|
dropout_1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout_1/dropout/Mul_1?
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes

:@*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Sigmoidn
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_5330

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@?  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5018

inputs8
*readvariableop_batch_normalization_2_gamma:09
+readvariableop_1_batch_normalization_2_beta:0O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:0U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????##0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????##02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????##0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????##0
 
_user_specified_nameinputs
?

?
?__inference_dense_layer_call_and_return_conditional_losses_5346

inputs7
"matmul_readvariableop_dense_kernel:???0
!biasadd_readvariableop_dense_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*!
_output_shapes
:???*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_5370

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2742

inputs8
*readvariableop_batch_normalization_3_gamma:@9
+readvariableop_1_batch_normalization_3_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
B
&__inference_dropout_layer_call_fn_5375

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_32012
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4982

inputs8
*readvariableop_batch_normalization_2_gamma:09
+readvariableop_1_batch_normalization_2_beta:0O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:0U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
8__inference_single_image_simple_model_layer_call_fn_4175
input_1'
conv2d_kernel:
conv2d_bias:'
batch_normalization_gamma:&
batch_normalization_beta:-
batch_normalization_moving_mean:1
#batch_normalization_moving_variance:)
conv2d_1_kernel: 
conv2d_1_bias: )
batch_normalization_1_gamma: (
batch_normalization_1_beta: /
!batch_normalization_1_moving_mean: 3
%batch_normalization_1_moving_variance: )
conv2d_2_kernel: 0
conv2d_2_bias:0)
batch_normalization_2_gamma:0(
batch_normalization_2_beta:0/
!batch_normalization_2_moving_mean:03
%batch_normalization_2_moving_variance:0)
conv2d_3_kernel:0@
conv2d_3_bias:@)
batch_normalization_3_gamma:@(
batch_normalization_3_beta:@/
!batch_normalization_3_moving_mean:@3
%batch_normalization_3_moving_variance:@)
conv2d_4_kernel:@@
conv2d_4_bias:@)
batch_normalization_4_gamma:@(
batch_normalization_4_beta:@/
!batch_normalization_4_moving_mean:@3
%batch_normalization_4_moving_variance:@!
dense_kernel:???

dense_bias:	?!
dense_1_kernel:	?@
dense_1_bias:@ 
dense_2_kernel:@
dense_2_bias:
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelconv2d_biasbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_varianceconv2d_1_kernelconv2d_1_biasbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_2_kernelconv2d_2_biasbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_varianceconv2d_3_kernelconv2d_3_biasbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_4_kernelconv2d_4_biasbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancedense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_39832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2706

inputs8
*readvariableop_batch_normalization_3_gamma:@9
+readvariableop_1_batch_normalization_3_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
B
&__inference_flatten_layer_call_fn_5335

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_31792
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_2994

inputs=
#conv2d_readvariableop_conv2d_kernel:0
"biasadd_readvariableop_conv2d_bias:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_conv2d_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4874

inputs8
*readvariableop_batch_normalization_1_gamma: 9
+readvariableop_1_batch_normalization_1_beta: O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean: U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
?__inference_dense_layer_call_and_return_conditional_losses_3192

inputs7
"matmul_readvariableop_dense_kernel:???0
!biasadd_readvariableop_dense_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*!
_output_shapes
:???*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5036

inputs8
*readvariableop_batch_normalization_2_gamma:09
+readvariableop_1_batch_normalization_2_beta:0O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:0U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????##0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????##02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????##0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????##0
 
_user_specified_nameinputs
??
?1
__inference__wrapped_model_2246
input_1^
Dsingle_image_simple_model_conv2d_conv2d_readvariableop_conv2d_kernel:Q
Csingle_image_simple_model_conv2d_biasadd_readvariableop_conv2d_bias:d
Vsingle_image_simple_model_batch_normalization_readvariableop_batch_normalization_gamma:e
Wsingle_image_simple_model_batch_normalization_readvariableop_1_batch_normalization_beta:{
msingle_image_simple_model_batch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:?
ssingle_image_simple_model_batch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:b
Hsingle_image_simple_model_conv2d_1_conv2d_readvariableop_conv2d_1_kernel: U
Gsingle_image_simple_model_conv2d_1_biasadd_readvariableop_conv2d_1_bias: h
Zsingle_image_simple_model_batch_normalization_1_readvariableop_batch_normalization_1_gamma: i
[single_image_simple_model_batch_normalization_1_readvariableop_1_batch_normalization_1_beta: 
qsingle_image_simple_model_batch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean: ?
wsingle_image_simple_model_batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance: b
Hsingle_image_simple_model_conv2d_2_conv2d_readvariableop_conv2d_2_kernel: 0U
Gsingle_image_simple_model_conv2d_2_biasadd_readvariableop_conv2d_2_bias:0h
Zsingle_image_simple_model_batch_normalization_2_readvariableop_batch_normalization_2_gamma:0i
[single_image_simple_model_batch_normalization_2_readvariableop_1_batch_normalization_2_beta:0
qsingle_image_simple_model_batch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:0?
wsingle_image_simple_model_batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:0b
Hsingle_image_simple_model_conv2d_3_conv2d_readvariableop_conv2d_3_kernel:0@U
Gsingle_image_simple_model_conv2d_3_biasadd_readvariableop_conv2d_3_bias:@h
Zsingle_image_simple_model_batch_normalization_3_readvariableop_batch_normalization_3_gamma:@i
[single_image_simple_model_batch_normalization_3_readvariableop_1_batch_normalization_3_beta:@
qsingle_image_simple_model_batch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:@?
wsingle_image_simple_model_batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:@b
Hsingle_image_simple_model_conv2d_4_conv2d_readvariableop_conv2d_4_kernel:@@U
Gsingle_image_simple_model_conv2d_4_biasadd_readvariableop_conv2d_4_bias:@h
Zsingle_image_simple_model_batch_normalization_4_readvariableop_batch_normalization_4_gamma:@i
[single_image_simple_model_batch_normalization_4_readvariableop_1_batch_normalization_4_beta:@
qsingle_image_simple_model_batch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:@?
wsingle_image_simple_model_batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:@W
Bsingle_image_simple_model_dense_matmul_readvariableop_dense_kernel:???P
Asingle_image_simple_model_dense_biasadd_readvariableop_dense_bias:	?Y
Fsingle_image_simple_model_dense_1_matmul_readvariableop_dense_1_kernel:	?@S
Esingle_image_simple_model_dense_1_biasadd_readvariableop_dense_1_bias:@X
Fsingle_image_simple_model_dense_2_matmul_readvariableop_dense_2_kernel:@S
Esingle_image_simple_model_dense_2_biasadd_readvariableop_dense_2_bias:
identity??Msingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOp?Osingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?<single_image_simple_model/batch_normalization/ReadVariableOp?>single_image_simple_model/batch_normalization/ReadVariableOp_1?Osingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Qsingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?>single_image_simple_model/batch_normalization_1/ReadVariableOp?@single_image_simple_model/batch_normalization_1/ReadVariableOp_1?Osingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Qsingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?>single_image_simple_model/batch_normalization_2/ReadVariableOp?@single_image_simple_model/batch_normalization_2/ReadVariableOp_1?Osingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Qsingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?>single_image_simple_model/batch_normalization_3/ReadVariableOp?@single_image_simple_model/batch_normalization_3/ReadVariableOp_1?Osingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Qsingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?>single_image_simple_model/batch_normalization_4/ReadVariableOp?@single_image_simple_model/batch_normalization_4/ReadVariableOp_1?7single_image_simple_model/conv2d/BiasAdd/ReadVariableOp?6single_image_simple_model/conv2d/Conv2D/ReadVariableOp?9single_image_simple_model/conv2d_1/BiasAdd/ReadVariableOp?8single_image_simple_model/conv2d_1/Conv2D/ReadVariableOp?9single_image_simple_model/conv2d_2/BiasAdd/ReadVariableOp?8single_image_simple_model/conv2d_2/Conv2D/ReadVariableOp?9single_image_simple_model/conv2d_3/BiasAdd/ReadVariableOp?8single_image_simple_model/conv2d_3/Conv2D/ReadVariableOp?9single_image_simple_model/conv2d_4/BiasAdd/ReadVariableOp?8single_image_simple_model/conv2d_4/Conv2D/ReadVariableOp?6single_image_simple_model/dense/BiasAdd/ReadVariableOp?5single_image_simple_model/dense/MatMul/ReadVariableOp?8single_image_simple_model/dense_1/BiasAdd/ReadVariableOp?7single_image_simple_model/dense_1/MatMul/ReadVariableOp?8single_image_simple_model/dense_2/BiasAdd/ReadVariableOp?7single_image_simple_model/dense_2/MatMul/ReadVariableOp?
6single_image_simple_model/conv2d/Conv2D/ReadVariableOpReadVariableOpDsingle_image_simple_model_conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:*
dtype028
6single_image_simple_model/conv2d/Conv2D/ReadVariableOp?
'single_image_simple_model/conv2d/Conv2DConv2Dinput_1>single_image_simple_model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2)
'single_image_simple_model/conv2d/Conv2D?
7single_image_simple_model/conv2d/BiasAdd/ReadVariableOpReadVariableOpCsingle_image_simple_model_conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:*
dtype029
7single_image_simple_model/conv2d/BiasAdd/ReadVariableOp?
(single_image_simple_model/conv2d/BiasAddBiasAdd0single_image_simple_model/conv2d/Conv2D:output:0?single_image_simple_model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2*
(single_image_simple_model/conv2d/BiasAdd?
%single_image_simple_model/conv2d/ReluRelu1single_image_simple_model/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2'
%single_image_simple_model/conv2d/Relu?
<single_image_simple_model/batch_normalization/ReadVariableOpReadVariableOpVsingle_image_simple_model_batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype02>
<single_image_simple_model/batch_normalization/ReadVariableOp?
>single_image_simple_model/batch_normalization/ReadVariableOp_1ReadVariableOpWsingle_image_simple_model_batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype02@
>single_image_simple_model/batch_normalization/ReadVariableOp_1?
Msingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpmsingle_image_simple_model_batch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype02O
Msingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Osingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpssingle_image_simple_model_batch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype02Q
Osingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
>single_image_simple_model/batch_normalization/FusedBatchNormV3FusedBatchNormV33single_image_simple_model/conv2d/Relu:activations:0Dsingle_image_simple_model/batch_normalization/ReadVariableOp:value:0Fsingle_image_simple_model/batch_normalization/ReadVariableOp_1:value:0Usingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Wsingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2@
>single_image_simple_model/batch_normalization/FusedBatchNormV3?
8single_image_simple_model/conv2d_1/Conv2D/ReadVariableOpReadVariableOpHsingle_image_simple_model_conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: *
dtype02:
8single_image_simple_model/conv2d_1/Conv2D/ReadVariableOp?
)single_image_simple_model/conv2d_1/Conv2DConv2DBsingle_image_simple_model/batch_normalization/FusedBatchNormV3:y:0@single_image_simple_model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH *
paddingVALID*
strides
2+
)single_image_simple_model/conv2d_1/Conv2D?
9single_image_simple_model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpGsingle_image_simple_model_conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
: *
dtype02;
9single_image_simple_model/conv2d_1/BiasAdd/ReadVariableOp?
*single_image_simple_model/conv2d_1/BiasAddBiasAdd2single_image_simple_model/conv2d_1/Conv2D:output:0Asingle_image_simple_model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH 2,
*single_image_simple_model/conv2d_1/BiasAdd?
'single_image_simple_model/conv2d_1/ReluRelu3single_image_simple_model/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????HH 2)
'single_image_simple_model/conv2d_1/Relu?
>single_image_simple_model/batch_normalization_1/ReadVariableOpReadVariableOpZsingle_image_simple_model_batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02@
>single_image_simple_model/batch_normalization_1/ReadVariableOp?
@single_image_simple_model/batch_normalization_1/ReadVariableOp_1ReadVariableOp[single_image_simple_model_batch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02B
@single_image_simple_model/batch_normalization_1/ReadVariableOp_1?
Osingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpqsingle_image_simple_model_batch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02Q
Osingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Qsingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpwsingle_image_simple_model_batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02S
Qsingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
@single_image_simple_model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV35single_image_simple_model/conv2d_1/Relu:activations:0Fsingle_image_simple_model/batch_normalization_1/ReadVariableOp:value:0Hsingle_image_simple_model/batch_normalization_1/ReadVariableOp_1:value:0Wsingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Ysingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????HH : : : : :*
epsilon%o?:*
is_training( 2B
@single_image_simple_model/batch_normalization_1/FusedBatchNormV3?
8single_image_simple_model/conv2d_2/Conv2D/ReadVariableOpReadVariableOpHsingle_image_simple_model_conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: 0*
dtype02:
8single_image_simple_model/conv2d_2/Conv2D/ReadVariableOp?
)single_image_simple_model/conv2d_2/Conv2DConv2DDsingle_image_simple_model/batch_normalization_1/FusedBatchNormV3:y:0@single_image_simple_model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##0*
paddingVALID*
strides
2+
)single_image_simple_model/conv2d_2/Conv2D?
9single_image_simple_model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpGsingle_image_simple_model_conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
:0*
dtype02;
9single_image_simple_model/conv2d_2/BiasAdd/ReadVariableOp?
*single_image_simple_model/conv2d_2/BiasAddBiasAdd2single_image_simple_model/conv2d_2/Conv2D:output:0Asingle_image_simple_model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##02,
*single_image_simple_model/conv2d_2/BiasAdd?
'single_image_simple_model/conv2d_2/ReluRelu3single_image_simple_model/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????##02)
'single_image_simple_model/conv2d_2/Relu?
>single_image_simple_model/batch_normalization_2/ReadVariableOpReadVariableOpZsingle_image_simple_model_batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
:0*
dtype02@
>single_image_simple_model/batch_normalization_2/ReadVariableOp?
@single_image_simple_model/batch_normalization_2/ReadVariableOp_1ReadVariableOp[single_image_simple_model_batch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
:0*
dtype02B
@single_image_simple_model/batch_normalization_2/ReadVariableOp_1?
Osingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpqsingle_image_simple_model_batch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02Q
Osingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Qsingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpwsingle_image_simple_model_batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02S
Qsingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
@single_image_simple_model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV35single_image_simple_model/conv2d_2/Relu:activations:0Fsingle_image_simple_model/batch_normalization_2/ReadVariableOp:value:0Hsingle_image_simple_model/batch_normalization_2/ReadVariableOp_1:value:0Wsingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Ysingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????##0:0:0:0:0:*
epsilon%o?:*
is_training( 2B
@single_image_simple_model/batch_normalization_2/FusedBatchNormV3?
8single_image_simple_model/conv2d_3/Conv2D/ReadVariableOpReadVariableOpHsingle_image_simple_model_conv2d_3_conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
:0@*
dtype02:
8single_image_simple_model/conv2d_3/Conv2D/ReadVariableOp?
)single_image_simple_model/conv2d_3/Conv2DConv2DDsingle_image_simple_model/batch_normalization_2/FusedBatchNormV3:y:0@single_image_simple_model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@*
paddingVALID*
strides
2+
)single_image_simple_model/conv2d_3/Conv2D?
9single_image_simple_model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpGsingle_image_simple_model_conv2d_3_biasadd_readvariableop_conv2d_3_bias*
_output_shapes
:@*
dtype02;
9single_image_simple_model/conv2d_3/BiasAdd/ReadVariableOp?
*single_image_simple_model/conv2d_3/BiasAddBiasAdd2single_image_simple_model/conv2d_3/Conv2D:output:0Asingle_image_simple_model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@2,
*single_image_simple_model/conv2d_3/BiasAdd?
'single_image_simple_model/conv2d_3/ReluRelu3single_image_simple_model/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!!@2)
'single_image_simple_model/conv2d_3/Relu?
>single_image_simple_model/batch_normalization_3/ReadVariableOpReadVariableOpZsingle_image_simple_model_batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
:@*
dtype02@
>single_image_simple_model/batch_normalization_3/ReadVariableOp?
@single_image_simple_model/batch_normalization_3/ReadVariableOp_1ReadVariableOp[single_image_simple_model_batch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
:@*
dtype02B
@single_image_simple_model/batch_normalization_3/ReadVariableOp_1?
Osingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpqsingle_image_simple_model_batch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:@*
dtype02Q
Osingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Qsingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpwsingle_image_simple_model_batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:@*
dtype02S
Qsingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
@single_image_simple_model/batch_normalization_3/FusedBatchNormV3FusedBatchNormV35single_image_simple_model/conv2d_3/Relu:activations:0Fsingle_image_simple_model/batch_normalization_3/ReadVariableOp:value:0Hsingle_image_simple_model/batch_normalization_3/ReadVariableOp_1:value:0Wsingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Ysingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????!!@:@:@:@:@:*
epsilon%o?:*
is_training( 2B
@single_image_simple_model/batch_normalization_3/FusedBatchNormV3?
8single_image_simple_model/conv2d_4/Conv2D/ReadVariableOpReadVariableOpHsingle_image_simple_model_conv2d_4_conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:@@*
dtype02:
8single_image_simple_model/conv2d_4/Conv2D/ReadVariableOp?
)single_image_simple_model/conv2d_4/Conv2DConv2DDsingle_image_simple_model/batch_normalization_3/FusedBatchNormV3:y:0@single_image_simple_model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2+
)single_image_simple_model/conv2d_4/Conv2D?
9single_image_simple_model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpGsingle_image_simple_model_conv2d_4_biasadd_readvariableop_conv2d_4_bias*
_output_shapes
:@*
dtype02;
9single_image_simple_model/conv2d_4/BiasAdd/ReadVariableOp?
*single_image_simple_model/conv2d_4/BiasAddBiasAdd2single_image_simple_model/conv2d_4/Conv2D:output:0Asingle_image_simple_model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2,
*single_image_simple_model/conv2d_4/BiasAdd?
'single_image_simple_model/conv2d_4/ReluRelu3single_image_simple_model/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2)
'single_image_simple_model/conv2d_4/Relu?
>single_image_simple_model/batch_normalization_4/ReadVariableOpReadVariableOpZsingle_image_simple_model_batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02@
>single_image_simple_model/batch_normalization_4/ReadVariableOp?
@single_image_simple_model/batch_normalization_4/ReadVariableOp_1ReadVariableOp[single_image_simple_model_batch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02B
@single_image_simple_model/batch_normalization_4/ReadVariableOp_1?
Osingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpqsingle_image_simple_model_batch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02Q
Osingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Qsingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpwsingle_image_simple_model_batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02S
Qsingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
@single_image_simple_model/batch_normalization_4/FusedBatchNormV3FusedBatchNormV35single_image_simple_model/conv2d_4/Relu:activations:0Fsingle_image_simple_model/batch_normalization_4/ReadVariableOp:value:0Hsingle_image_simple_model/batch_normalization_4/ReadVariableOp_1:value:0Wsingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Ysingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2B
@single_image_simple_model/batch_normalization_4/FusedBatchNormV3?
'single_image_simple_model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@?  2)
'single_image_simple_model/flatten/Const?
)single_image_simple_model/flatten/ReshapeReshapeDsingle_image_simple_model/batch_normalization_4/FusedBatchNormV3:y:00single_image_simple_model/flatten/Const:output:0*
T0*)
_output_shapes
:???????????2+
)single_image_simple_model/flatten/Reshape?
5single_image_simple_model/dense/MatMul/ReadVariableOpReadVariableOpBsingle_image_simple_model_dense_matmul_readvariableop_dense_kernel*!
_output_shapes
:???*
dtype027
5single_image_simple_model/dense/MatMul/ReadVariableOp?
&single_image_simple_model/dense/MatMulMatMul2single_image_simple_model/flatten/Reshape:output:0=single_image_simple_model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&single_image_simple_model/dense/MatMul?
6single_image_simple_model/dense/BiasAdd/ReadVariableOpReadVariableOpAsingle_image_simple_model_dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype028
6single_image_simple_model/dense/BiasAdd/ReadVariableOp?
'single_image_simple_model/dense/BiasAddBiasAdd0single_image_simple_model/dense/MatMul:product:0>single_image_simple_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'single_image_simple_model/dense/BiasAdd?
$single_image_simple_model/dense/ReluRelu0single_image_simple_model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2&
$single_image_simple_model/dense/Relu?
*single_image_simple_model/dropout/IdentityIdentity2single_image_simple_model/dense/Relu:activations:0*
T0*(
_output_shapes
:??????????2,
*single_image_simple_model/dropout/Identity?
7single_image_simple_model/dense_1/MatMul/ReadVariableOpReadVariableOpFsingle_image_simple_model_dense_1_matmul_readvariableop_dense_1_kernel*
_output_shapes
:	?@*
dtype029
7single_image_simple_model/dense_1/MatMul/ReadVariableOp?
(single_image_simple_model/dense_1/MatMulMatMul3single_image_simple_model/dropout/Identity:output:0?single_image_simple_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2*
(single_image_simple_model/dense_1/MatMul?
8single_image_simple_model/dense_1/BiasAdd/ReadVariableOpReadVariableOpEsingle_image_simple_model_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:@*
dtype02:
8single_image_simple_model/dense_1/BiasAdd/ReadVariableOp?
)single_image_simple_model/dense_1/BiasAddBiasAdd2single_image_simple_model/dense_1/MatMul:product:0@single_image_simple_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2+
)single_image_simple_model/dense_1/BiasAdd?
&single_image_simple_model/dense_1/ReluRelu2single_image_simple_model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2(
&single_image_simple_model/dense_1/Relu?
,single_image_simple_model/dropout_1/IdentityIdentity4single_image_simple_model/dense_1/Relu:activations:0*
T0*'
_output_shapes
:?????????@2.
,single_image_simple_model/dropout_1/Identity?
7single_image_simple_model/dense_2/MatMul/ReadVariableOpReadVariableOpFsingle_image_simple_model_dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes

:@*
dtype029
7single_image_simple_model/dense_2/MatMul/ReadVariableOp?
(single_image_simple_model/dense_2/MatMulMatMul5single_image_simple_model/dropout_1/Identity:output:0?single_image_simple_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(single_image_simple_model/dense_2/MatMul?
8single_image_simple_model/dense_2/BiasAdd/ReadVariableOpReadVariableOpEsingle_image_simple_model_dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02:
8single_image_simple_model/dense_2/BiasAdd/ReadVariableOp?
)single_image_simple_model/dense_2/BiasAddBiasAdd2single_image_simple_model/dense_2/MatMul:product:0@single_image_simple_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)single_image_simple_model/dense_2/BiasAdd?
)single_image_simple_model/dense_2/SigmoidSigmoid2single_image_simple_model/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2+
)single_image_simple_model/dense_2/Sigmoid?
IdentityIdentity-single_image_simple_model/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOpN^single_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOpP^single_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1=^single_image_simple_model/batch_normalization/ReadVariableOp?^single_image_simple_model/batch_normalization/ReadVariableOp_1P^single_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpR^single_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?^single_image_simple_model/batch_normalization_1/ReadVariableOpA^single_image_simple_model/batch_normalization_1/ReadVariableOp_1P^single_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpR^single_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?^single_image_simple_model/batch_normalization_2/ReadVariableOpA^single_image_simple_model/batch_normalization_2/ReadVariableOp_1P^single_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpR^single_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?^single_image_simple_model/batch_normalization_3/ReadVariableOpA^single_image_simple_model/batch_normalization_3/ReadVariableOp_1P^single_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpR^single_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?^single_image_simple_model/batch_normalization_4/ReadVariableOpA^single_image_simple_model/batch_normalization_4/ReadVariableOp_18^single_image_simple_model/conv2d/BiasAdd/ReadVariableOp7^single_image_simple_model/conv2d/Conv2D/ReadVariableOp:^single_image_simple_model/conv2d_1/BiasAdd/ReadVariableOp9^single_image_simple_model/conv2d_1/Conv2D/ReadVariableOp:^single_image_simple_model/conv2d_2/BiasAdd/ReadVariableOp9^single_image_simple_model/conv2d_2/Conv2D/ReadVariableOp:^single_image_simple_model/conv2d_3/BiasAdd/ReadVariableOp9^single_image_simple_model/conv2d_3/Conv2D/ReadVariableOp:^single_image_simple_model/conv2d_4/BiasAdd/ReadVariableOp9^single_image_simple_model/conv2d_4/Conv2D/ReadVariableOp7^single_image_simple_model/dense/BiasAdd/ReadVariableOp6^single_image_simple_model/dense/MatMul/ReadVariableOp9^single_image_simple_model/dense_1/BiasAdd/ReadVariableOp8^single_image_simple_model/dense_1/MatMul/ReadVariableOp9^single_image_simple_model/dense_2/BiasAdd/ReadVariableOp8^single_image_simple_model/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Msingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOpMsingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Osingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Osingle_image_simple_model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12|
<single_image_simple_model/batch_normalization/ReadVariableOp<single_image_simple_model/batch_normalization/ReadVariableOp2?
>single_image_simple_model/batch_normalization/ReadVariableOp_1>single_image_simple_model/batch_normalization/ReadVariableOp_12?
Osingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpOsingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Qsingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Qsingle_image_simple_model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12?
>single_image_simple_model/batch_normalization_1/ReadVariableOp>single_image_simple_model/batch_normalization_1/ReadVariableOp2?
@single_image_simple_model/batch_normalization_1/ReadVariableOp_1@single_image_simple_model/batch_normalization_1/ReadVariableOp_12?
Osingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpOsingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Qsingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Qsingle_image_simple_model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12?
>single_image_simple_model/batch_normalization_2/ReadVariableOp>single_image_simple_model/batch_normalization_2/ReadVariableOp2?
@single_image_simple_model/batch_normalization_2/ReadVariableOp_1@single_image_simple_model/batch_normalization_2/ReadVariableOp_12?
Osingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpOsingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Qsingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Qsingle_image_simple_model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12?
>single_image_simple_model/batch_normalization_3/ReadVariableOp>single_image_simple_model/batch_normalization_3/ReadVariableOp2?
@single_image_simple_model/batch_normalization_3/ReadVariableOp_1@single_image_simple_model/batch_normalization_3/ReadVariableOp_12?
Osingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpOsingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Qsingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Qsingle_image_simple_model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12?
>single_image_simple_model/batch_normalization_4/ReadVariableOp>single_image_simple_model/batch_normalization_4/ReadVariableOp2?
@single_image_simple_model/batch_normalization_4/ReadVariableOp_1@single_image_simple_model/batch_normalization_4/ReadVariableOp_12r
7single_image_simple_model/conv2d/BiasAdd/ReadVariableOp7single_image_simple_model/conv2d/BiasAdd/ReadVariableOp2p
6single_image_simple_model/conv2d/Conv2D/ReadVariableOp6single_image_simple_model/conv2d/Conv2D/ReadVariableOp2v
9single_image_simple_model/conv2d_1/BiasAdd/ReadVariableOp9single_image_simple_model/conv2d_1/BiasAdd/ReadVariableOp2t
8single_image_simple_model/conv2d_1/Conv2D/ReadVariableOp8single_image_simple_model/conv2d_1/Conv2D/ReadVariableOp2v
9single_image_simple_model/conv2d_2/BiasAdd/ReadVariableOp9single_image_simple_model/conv2d_2/BiasAdd/ReadVariableOp2t
8single_image_simple_model/conv2d_2/Conv2D/ReadVariableOp8single_image_simple_model/conv2d_2/Conv2D/ReadVariableOp2v
9single_image_simple_model/conv2d_3/BiasAdd/ReadVariableOp9single_image_simple_model/conv2d_3/BiasAdd/ReadVariableOp2t
8single_image_simple_model/conv2d_3/Conv2D/ReadVariableOp8single_image_simple_model/conv2d_3/Conv2D/ReadVariableOp2v
9single_image_simple_model/conv2d_4/BiasAdd/ReadVariableOp9single_image_simple_model/conv2d_4/BiasAdd/ReadVariableOp2t
8single_image_simple_model/conv2d_4/Conv2D/ReadVariableOp8single_image_simple_model/conv2d_4/Conv2D/ReadVariableOp2p
6single_image_simple_model/dense/BiasAdd/ReadVariableOp6single_image_simple_model/dense/BiasAdd/ReadVariableOp2n
5single_image_simple_model/dense/MatMul/ReadVariableOp5single_image_simple_model/dense/MatMul/ReadVariableOp2t
8single_image_simple_model/dense_1/BiasAdd/ReadVariableOp8single_image_simple_model/dense_1/BiasAdd/ReadVariableOp2r
7single_image_simple_model/dense_1/MatMul/ReadVariableOp7single_image_simple_model/dense_1/MatMul/ReadVariableOp2t
8single_image_simple_model/dense_2/BiasAdd/ReadVariableOp8single_image_simple_model/dense_2/BiasAdd/ReadVariableOp2r
7single_image_simple_model/dense_2/MatMul/ReadVariableOp7single_image_simple_model/dense_2/MatMul/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
$__inference_dense_layer_call_fn_5353

inputs!
dense_kernel:???

dense_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_31922
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
Σ
?B
 __inference__traced_restore_6046
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias::
,assignvariableop_2_batch_normalization_gamma:9
+assignvariableop_3_batch_normalization_beta:@
2assignvariableop_4_batch_normalization_moving_mean:D
6assignvariableop_5_batch_normalization_moving_variance:<
"assignvariableop_6_conv2d_1_kernel: .
 assignvariableop_7_conv2d_1_bias: <
.assignvariableop_8_batch_normalization_1_gamma: ;
-assignvariableop_9_batch_normalization_1_beta: C
5assignvariableop_10_batch_normalization_1_moving_mean: G
9assignvariableop_11_batch_normalization_1_moving_variance: =
#assignvariableop_12_conv2d_2_kernel: 0/
!assignvariableop_13_conv2d_2_bias:0=
/assignvariableop_14_batch_normalization_2_gamma:0<
.assignvariableop_15_batch_normalization_2_beta:0C
5assignvariableop_16_batch_normalization_2_moving_mean:0G
9assignvariableop_17_batch_normalization_2_moving_variance:0=
#assignvariableop_18_conv2d_3_kernel:0@/
!assignvariableop_19_conv2d_3_bias:@=
/assignvariableop_20_batch_normalization_3_gamma:@<
.assignvariableop_21_batch_normalization_3_beta:@C
5assignvariableop_22_batch_normalization_3_moving_mean:@G
9assignvariableop_23_batch_normalization_3_moving_variance:@=
#assignvariableop_24_conv2d_4_kernel:@@/
!assignvariableop_25_conv2d_4_bias:@=
/assignvariableop_26_batch_normalization_4_gamma:@<
.assignvariableop_27_batch_normalization_4_beta:@C
5assignvariableop_28_batch_normalization_4_moving_mean:@G
9assignvariableop_29_batch_normalization_4_moving_variance:@5
 assignvariableop_30_dense_kernel:???-
assignvariableop_31_dense_bias:	?5
"assignvariableop_32_dense_1_kernel:	?@.
 assignvariableop_33_dense_1_bias:@4
"assignvariableop_34_dense_2_kernel:@.
 assignvariableop_35_dense_2_bias:0
&assignvariableop_36_training_adam_iter:	 2
(assignvariableop_37_training_adam_beta_1: 2
(assignvariableop_38_training_adam_beta_2: 1
'assignvariableop_39_training_adam_decay: 9
/assignvariableop_40_training_adam_learning_rate: #
assignvariableop_41_total: #
assignvariableop_42_count: K
1assignvariableop_43_training_adam_conv2d_kernel_m:=
/assignvariableop_44_training_adam_conv2d_bias_m:K
=assignvariableop_45_training_adam_batch_normalization_gamma_m:J
<assignvariableop_46_training_adam_batch_normalization_beta_m:M
3assignvariableop_47_training_adam_conv2d_1_kernel_m: ?
1assignvariableop_48_training_adam_conv2d_1_bias_m: M
?assignvariableop_49_training_adam_batch_normalization_1_gamma_m: L
>assignvariableop_50_training_adam_batch_normalization_1_beta_m: M
3assignvariableop_51_training_adam_conv2d_2_kernel_m: 0?
1assignvariableop_52_training_adam_conv2d_2_bias_m:0M
?assignvariableop_53_training_adam_batch_normalization_2_gamma_m:0L
>assignvariableop_54_training_adam_batch_normalization_2_beta_m:0M
3assignvariableop_55_training_adam_conv2d_3_kernel_m:0@?
1assignvariableop_56_training_adam_conv2d_3_bias_m:@M
?assignvariableop_57_training_adam_batch_normalization_3_gamma_m:@L
>assignvariableop_58_training_adam_batch_normalization_3_beta_m:@M
3assignvariableop_59_training_adam_conv2d_4_kernel_m:@@?
1assignvariableop_60_training_adam_conv2d_4_bias_m:@M
?assignvariableop_61_training_adam_batch_normalization_4_gamma_m:@L
>assignvariableop_62_training_adam_batch_normalization_4_beta_m:@E
0assignvariableop_63_training_adam_dense_kernel_m:???=
.assignvariableop_64_training_adam_dense_bias_m:	?E
2assignvariableop_65_training_adam_dense_1_kernel_m:	?@>
0assignvariableop_66_training_adam_dense_1_bias_m:@D
2assignvariableop_67_training_adam_dense_2_kernel_m:@>
0assignvariableop_68_training_adam_dense_2_bias_m:K
1assignvariableop_69_training_adam_conv2d_kernel_v:=
/assignvariableop_70_training_adam_conv2d_bias_v:K
=assignvariableop_71_training_adam_batch_normalization_gamma_v:J
<assignvariableop_72_training_adam_batch_normalization_beta_v:M
3assignvariableop_73_training_adam_conv2d_1_kernel_v: ?
1assignvariableop_74_training_adam_conv2d_1_bias_v: M
?assignvariableop_75_training_adam_batch_normalization_1_gamma_v: L
>assignvariableop_76_training_adam_batch_normalization_1_beta_v: M
3assignvariableop_77_training_adam_conv2d_2_kernel_v: 0?
1assignvariableop_78_training_adam_conv2d_2_bias_v:0M
?assignvariableop_79_training_adam_batch_normalization_2_gamma_v:0L
>assignvariableop_80_training_adam_batch_normalization_2_beta_v:0M
3assignvariableop_81_training_adam_conv2d_3_kernel_v:0@?
1assignvariableop_82_training_adam_conv2d_3_bias_v:@M
?assignvariableop_83_training_adam_batch_normalization_3_gamma_v:@L
>assignvariableop_84_training_adam_batch_normalization_3_beta_v:@M
3assignvariableop_85_training_adam_conv2d_4_kernel_v:@@?
1assignvariableop_86_training_adam_conv2d_4_bias_v:@M
?assignvariableop_87_training_adam_batch_normalization_4_gamma_v:@L
>assignvariableop_88_training_adam_batch_normalization_4_beta_v:@E
0assignvariableop_89_training_adam_dense_kernel_v:???=
.assignvariableop_90_training_adam_dense_bias_v:	?E
2assignvariableop_91_training_adam_dense_1_kernel_v:	?@>
0assignvariableop_92_training_adam_dense_1_bias_v:@D
2assignvariableop_93_training_adam_dense_2_kernel_v:@>
0assignvariableop_94_training_adam_dense_2_bias_v:
identity_96??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?5
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:`*
dtype0*?4
value?4B?4`B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:`*
dtype0*?
value?B?`B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*n
dtypesd
b2`	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp assignvariableop_30_dense_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_dense_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense_1_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense_1_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_2_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp assignvariableop_35_dense_2_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_training_adam_iterIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_training_adam_beta_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_training_adam_beta_2Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp'assignvariableop_39_training_adam_decayIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp/assignvariableop_40_training_adam_learning_rateIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp1assignvariableop_43_training_adam_conv2d_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp/assignvariableop_44_training_adam_conv2d_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp=assignvariableop_45_training_adam_batch_normalization_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp<assignvariableop_46_training_adam_batch_normalization_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp3assignvariableop_47_training_adam_conv2d_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp1assignvariableop_48_training_adam_conv2d_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp?assignvariableop_49_training_adam_batch_normalization_1_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp>assignvariableop_50_training_adam_batch_normalization_1_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp3assignvariableop_51_training_adam_conv2d_2_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp1assignvariableop_52_training_adam_conv2d_2_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp?assignvariableop_53_training_adam_batch_normalization_2_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp>assignvariableop_54_training_adam_batch_normalization_2_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp3assignvariableop_55_training_adam_conv2d_3_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp1assignvariableop_56_training_adam_conv2d_3_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp?assignvariableop_57_training_adam_batch_normalization_3_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp>assignvariableop_58_training_adam_batch_normalization_3_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp3assignvariableop_59_training_adam_conv2d_4_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp1assignvariableop_60_training_adam_conv2d_4_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp?assignvariableop_61_training_adam_batch_normalization_4_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp>assignvariableop_62_training_adam_batch_normalization_4_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp0assignvariableop_63_training_adam_dense_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp.assignvariableop_64_training_adam_dense_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp2assignvariableop_65_training_adam_dense_1_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp0assignvariableop_66_training_adam_dense_1_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp2assignvariableop_67_training_adam_dense_2_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp0assignvariableop_68_training_adam_dense_2_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp1assignvariableop_69_training_adam_conv2d_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp/assignvariableop_70_training_adam_conv2d_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp=assignvariableop_71_training_adam_batch_normalization_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp<assignvariableop_72_training_adam_batch_normalization_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp3assignvariableop_73_training_adam_conv2d_1_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp1assignvariableop_74_training_adam_conv2d_1_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp?assignvariableop_75_training_adam_batch_normalization_1_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp>assignvariableop_76_training_adam_batch_normalization_1_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp3assignvariableop_77_training_adam_conv2d_2_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp1assignvariableop_78_training_adam_conv2d_2_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp?assignvariableop_79_training_adam_batch_normalization_2_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp>assignvariableop_80_training_adam_batch_normalization_2_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp3assignvariableop_81_training_adam_conv2d_3_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp1assignvariableop_82_training_adam_conv2d_3_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp?assignvariableop_83_training_adam_batch_normalization_3_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp>assignvariableop_84_training_adam_batch_normalization_3_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp3assignvariableop_85_training_adam_conv2d_4_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp1assignvariableop_86_training_adam_conv2d_4_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp?assignvariableop_87_training_adam_batch_normalization_4_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp>assignvariableop_88_training_adam_batch_normalization_4_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp0assignvariableop_89_training_adam_dense_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp.assignvariableop_90_training_adam_dense_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp2assignvariableop_91_training_adam_dense_1_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp0assignvariableop_92_training_adam_dense_1_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp2assignvariableop_93_training_adam_dense_2_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp0assignvariableop_94_training_adam_dense_2_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_949
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_95Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_95f
Identity_96IdentityIdentity_95:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_96?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_96Identity_96:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_94:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4748

inputs6
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_5415

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_2_layer_call_fn_5063

inputs)
batch_normalization_2_gamma:0(
batch_normalization_2_beta:0/
!batch_normalization_2_moving_mean:03
%batch_normalization_2_moving_variance:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????##0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_30912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????##02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????##0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????##0
 
_user_specified_nameinputs
?
?
'__inference_conv2d_1_layer_call_fn_4838

inputs)
conv2d_1_kernel: 
conv2d_1_bias: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_kernelconv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_30322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????HH 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5270

inputs8
*readvariableop_batch_normalization_4_gamma:@9
+readvariableop_1_batch_normalization_4_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2268

inputs6
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_3319

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3731

inputs8
*readvariableop_batch_normalization_1_gamma: 9
+readvariableop_1_batch_normalization_1_beta: O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean: U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????HH : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????HH 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????HH : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????HH 
 
_user_specified_nameinputs
?

?
2__inference_batch_normalization_layer_call_fn_4793

inputs'
batch_normalization_gamma:&
batch_normalization_beta:-
batch_normalization_moving_mean:1
#batch_normalization_moving_variance:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_22682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
4__inference_batch_normalization_2_layer_call_fn_5045

inputs)
batch_normalization_2_gamma:0(
batch_normalization_2_beta:0/
!batch_normalization_2_moving_mean:03
%batch_normalization_2_moving_variance:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_25602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5209

inputs?
%conv2d_readvariableop_conv2d_4_kernel:@@2
$biasadd_readvariableop_conv2d_4_bias:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_4_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*2
_input_shapes!
:?????????!!@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????!!@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2888

inputs8
*readvariableop_batch_normalization_4_gamma:@9
+readvariableop_1_batch_normalization_4_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
4__inference_batch_normalization_4_layer_call_fn_5297

inputs)
batch_normalization_4_gamma:@(
batch_normalization_4_beta:@/
!batch_normalization_4_moving_mean:@3
%batch_normalization_4_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_28522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4892

inputs8
*readvariableop_batch_normalization_1_gamma: 9
+readvariableop_1_batch_normalization_1_beta: O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean: U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????HH : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????HH 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????HH : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????HH 
 
_user_specified_nameinputs
?

?
4__inference_batch_normalization_3_layer_call_fn_5171

inputs)
batch_normalization_3_gamma:@(
batch_normalization_3_beta:@/
!batch_normalization_3_moving_mean:@3
%batch_normalization_3_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5108

inputs8
*readvariableop_batch_normalization_3_gamma:@9
+readvariableop_1_batch_normalization_3_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5162

inputs8
*readvariableop_batch_normalization_3_gamma:@9
+readvariableop_1_batch_normalization_3_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????!!@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????!!@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????!!@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????!!@
 
_user_specified_nameinputs
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_3378

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4784

inputs6
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4766

inputs6
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_dense_1_layer_call_and_return_conditional_losses_3214

inputs7
$matmul_readvariableop_dense_1_kernel:	?@1
#biasadd_readvariableop_dense_1_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
4__inference_batch_normalization_1_layer_call_fn_4919

inputs)
batch_normalization_1_gamma: (
batch_normalization_1_beta: /
!batch_normalization_1_moving_mean: 3
%batch_normalization_1_moving_variance: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_3146

inputs?
%conv2d_readvariableop_conv2d_4_kernel:@@2
$biasadd_readvariableop_conv2d_4_bias:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_4_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????!!@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????!!@
 
_user_specified_nameinputs
?

?
4__inference_batch_normalization_1_layer_call_fn_4928

inputs)
batch_normalization_1_gamma: (
batch_normalization_1_beta: /
!batch_normalization_1_moving_mean: 3
%batch_normalization_1_moving_variance: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5000

inputs8
*readvariableop_batch_normalization_2_gamma:09
+readvariableop_1_batch_normalization_2_beta:0O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:0U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?

?
4__inference_batch_normalization_2_layer_call_fn_5054

inputs)
batch_normalization_2_gamma:0(
batch_normalization_2_beta:0/
!batch_normalization_2_moving_mean:03
%batch_normalization_2_moving_variance:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_25962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_4_layer_call_fn_5324

inputs)
batch_normalization_4_gamma:@(
batch_normalization_4_beta:@/
!batch_normalization_4_moving_mean:@3
%batch_normalization_4_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?.
__inference__traced_save_5751
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_training_adam_conv2d_kernel_m_read_readvariableop:
6savev2_training_adam_conv2d_bias_m_read_readvariableopH
Dsavev2_training_adam_batch_normalization_gamma_m_read_readvariableopG
Csavev2_training_adam_batch_normalization_beta_m_read_readvariableop>
:savev2_training_adam_conv2d_1_kernel_m_read_readvariableop<
8savev2_training_adam_conv2d_1_bias_m_read_readvariableopJ
Fsavev2_training_adam_batch_normalization_1_gamma_m_read_readvariableopI
Esavev2_training_adam_batch_normalization_1_beta_m_read_readvariableop>
:savev2_training_adam_conv2d_2_kernel_m_read_readvariableop<
8savev2_training_adam_conv2d_2_bias_m_read_readvariableopJ
Fsavev2_training_adam_batch_normalization_2_gamma_m_read_readvariableopI
Esavev2_training_adam_batch_normalization_2_beta_m_read_readvariableop>
:savev2_training_adam_conv2d_3_kernel_m_read_readvariableop<
8savev2_training_adam_conv2d_3_bias_m_read_readvariableopJ
Fsavev2_training_adam_batch_normalization_3_gamma_m_read_readvariableopI
Esavev2_training_adam_batch_normalization_3_beta_m_read_readvariableop>
:savev2_training_adam_conv2d_4_kernel_m_read_readvariableop<
8savev2_training_adam_conv2d_4_bias_m_read_readvariableopJ
Fsavev2_training_adam_batch_normalization_4_gamma_m_read_readvariableopI
Esavev2_training_adam_batch_normalization_4_beta_m_read_readvariableop;
7savev2_training_adam_dense_kernel_m_read_readvariableop9
5savev2_training_adam_dense_bias_m_read_readvariableop=
9savev2_training_adam_dense_1_kernel_m_read_readvariableop;
7savev2_training_adam_dense_1_bias_m_read_readvariableop=
9savev2_training_adam_dense_2_kernel_m_read_readvariableop;
7savev2_training_adam_dense_2_bias_m_read_readvariableop<
8savev2_training_adam_conv2d_kernel_v_read_readvariableop:
6savev2_training_adam_conv2d_bias_v_read_readvariableopH
Dsavev2_training_adam_batch_normalization_gamma_v_read_readvariableopG
Csavev2_training_adam_batch_normalization_beta_v_read_readvariableop>
:savev2_training_adam_conv2d_1_kernel_v_read_readvariableop<
8savev2_training_adam_conv2d_1_bias_v_read_readvariableopJ
Fsavev2_training_adam_batch_normalization_1_gamma_v_read_readvariableopI
Esavev2_training_adam_batch_normalization_1_beta_v_read_readvariableop>
:savev2_training_adam_conv2d_2_kernel_v_read_readvariableop<
8savev2_training_adam_conv2d_2_bias_v_read_readvariableopJ
Fsavev2_training_adam_batch_normalization_2_gamma_v_read_readvariableopI
Esavev2_training_adam_batch_normalization_2_beta_v_read_readvariableop>
:savev2_training_adam_conv2d_3_kernel_v_read_readvariableop<
8savev2_training_adam_conv2d_3_bias_v_read_readvariableopJ
Fsavev2_training_adam_batch_normalization_3_gamma_v_read_readvariableopI
Esavev2_training_adam_batch_normalization_3_beta_v_read_readvariableop>
:savev2_training_adam_conv2d_4_kernel_v_read_readvariableop<
8savev2_training_adam_conv2d_4_bias_v_read_readvariableopJ
Fsavev2_training_adam_batch_normalization_4_gamma_v_read_readvariableopI
Esavev2_training_adam_batch_normalization_4_beta_v_read_readvariableop;
7savev2_training_adam_dense_kernel_v_read_readvariableop9
5savev2_training_adam_dense_bias_v_read_readvariableop=
9savev2_training_adam_dense_1_kernel_v_read_readvariableop;
7savev2_training_adam_dense_1_bias_v_read_readvariableop=
9savev2_training_adam_dense_2_kernel_v_read_readvariableop;
7savev2_training_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?5
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:`*
dtype0*?4
value?4B?4`B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:`*
dtype0*?
value?B?`B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?,
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_training_adam_conv2d_kernel_m_read_readvariableop6savev2_training_adam_conv2d_bias_m_read_readvariableopDsavev2_training_adam_batch_normalization_gamma_m_read_readvariableopCsavev2_training_adam_batch_normalization_beta_m_read_readvariableop:savev2_training_adam_conv2d_1_kernel_m_read_readvariableop8savev2_training_adam_conv2d_1_bias_m_read_readvariableopFsavev2_training_adam_batch_normalization_1_gamma_m_read_readvariableopEsavev2_training_adam_batch_normalization_1_beta_m_read_readvariableop:savev2_training_adam_conv2d_2_kernel_m_read_readvariableop8savev2_training_adam_conv2d_2_bias_m_read_readvariableopFsavev2_training_adam_batch_normalization_2_gamma_m_read_readvariableopEsavev2_training_adam_batch_normalization_2_beta_m_read_readvariableop:savev2_training_adam_conv2d_3_kernel_m_read_readvariableop8savev2_training_adam_conv2d_3_bias_m_read_readvariableopFsavev2_training_adam_batch_normalization_3_gamma_m_read_readvariableopEsavev2_training_adam_batch_normalization_3_beta_m_read_readvariableop:savev2_training_adam_conv2d_4_kernel_m_read_readvariableop8savev2_training_adam_conv2d_4_bias_m_read_readvariableopFsavev2_training_adam_batch_normalization_4_gamma_m_read_readvariableopEsavev2_training_adam_batch_normalization_4_beta_m_read_readvariableop7savev2_training_adam_dense_kernel_m_read_readvariableop5savev2_training_adam_dense_bias_m_read_readvariableop9savev2_training_adam_dense_1_kernel_m_read_readvariableop7savev2_training_adam_dense_1_bias_m_read_readvariableop9savev2_training_adam_dense_2_kernel_m_read_readvariableop7savev2_training_adam_dense_2_bias_m_read_readvariableop8savev2_training_adam_conv2d_kernel_v_read_readvariableop6savev2_training_adam_conv2d_bias_v_read_readvariableopDsavev2_training_adam_batch_normalization_gamma_v_read_readvariableopCsavev2_training_adam_batch_normalization_beta_v_read_readvariableop:savev2_training_adam_conv2d_1_kernel_v_read_readvariableop8savev2_training_adam_conv2d_1_bias_v_read_readvariableopFsavev2_training_adam_batch_normalization_1_gamma_v_read_readvariableopEsavev2_training_adam_batch_normalization_1_beta_v_read_readvariableop:savev2_training_adam_conv2d_2_kernel_v_read_readvariableop8savev2_training_adam_conv2d_2_bias_v_read_readvariableopFsavev2_training_adam_batch_normalization_2_gamma_v_read_readvariableopEsavev2_training_adam_batch_normalization_2_beta_v_read_readvariableop:savev2_training_adam_conv2d_3_kernel_v_read_readvariableop8savev2_training_adam_conv2d_3_bias_v_read_readvariableopFsavev2_training_adam_batch_normalization_3_gamma_v_read_readvariableopEsavev2_training_adam_batch_normalization_3_beta_v_read_readvariableop:savev2_training_adam_conv2d_4_kernel_v_read_readvariableop8savev2_training_adam_conv2d_4_bias_v_read_readvariableopFsavev2_training_adam_batch_normalization_4_gamma_v_read_readvariableopEsavev2_training_adam_batch_normalization_4_beta_v_read_readvariableop7savev2_training_adam_dense_kernel_v_read_readvariableop5savev2_training_adam_dense_bias_v_read_readvariableop9savev2_training_adam_dense_1_kernel_v_read_readvariableop7savev2_training_adam_dense_1_bias_v_read_readvariableop9savev2_training_adam_dense_2_kernel_v_read_readvariableop7savev2_training_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *n
dtypesd
b2`	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::: : : : : : : 0:0:0:0:0:0:0@:@:@:@:@:@:@@:@:@:@:@:@:???:?:	?@:@:@:: : : : : : : ::::: : : : : 0:0:0:0:0@:@:@:@:@@:@:@:@:???:?:	?@:@:@:::::: : : : : 0:0:0:0:0@:@:@:@:@@:@:@:@:???:?:	?@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:'#
!
_output_shapes
:???:! 

_output_shapes	
:?:%!!

_output_shapes
:	?@: "

_output_shapes
:@:$# 

_output_shapes

:@: $

_output_shapes
::%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,,(
&
_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: :,4(
&
_output_shapes
: 0: 5

_output_shapes
:0: 6

_output_shapes
:0: 7

_output_shapes
:0:,8(
&
_output_shapes
:0@: 9

_output_shapes
:@: :

_output_shapes
:@: ;

_output_shapes
:@:,<(
&
_output_shapes
:@@: =

_output_shapes
:@: >

_output_shapes
:@: ?

_output_shapes
:@:'@#
!
_output_shapes
:???:!A

_output_shapes	
:?:%B!

_output_shapes
:	?@: C

_output_shapes
:@:$D 

_output_shapes

:@: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
:: H

_output_shapes
:: I

_output_shapes
::,J(
&
_output_shapes
: : K

_output_shapes
: : L

_output_shapes
: : M

_output_shapes
: :,N(
&
_output_shapes
: 0: O

_output_shapes
:0: P

_output_shapes
:0: Q

_output_shapes
:0:,R(
&
_output_shapes
:0@: S

_output_shapes
:@: T

_output_shapes
:@: U

_output_shapes
:@:,V(
&
_output_shapes
:@@: W

_output_shapes
:@: X

_output_shapes
:@: Y

_output_shapes
:@:'Z#
!
_output_shapes
:???:![

_output_shapes	
:?:%\!

_output_shapes
:	?@: ]

_output_shapes
:@:$^ 

_output_shapes

:@: _

_output_shapes
::`

_output_shapes
: 
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_3223

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
8__inference_single_image_simple_model_layer_call_fn_4694

inputs'
conv2d_kernel:
conv2d_bias:'
batch_normalization_gamma:&
batch_normalization_beta:-
batch_normalization_moving_mean:1
#batch_normalization_moving_variance:)
conv2d_1_kernel: 
conv2d_1_bias: )
batch_normalization_1_gamma: (
batch_normalization_1_beta: /
!batch_normalization_1_moving_mean: 3
%batch_normalization_1_moving_variance: )
conv2d_2_kernel: 0
conv2d_2_bias:0)
batch_normalization_2_gamma:0(
batch_normalization_2_beta:0/
!batch_normalization_2_moving_mean:03
%batch_normalization_2_moving_variance:0)
conv2d_3_kernel:0@
conv2d_3_bias:@)
batch_normalization_3_gamma:@(
batch_normalization_3_beta:@/
!batch_normalization_3_moving_mean:@3
%batch_normalization_3_moving_variance:@)
conv2d_4_kernel:@@
conv2d_4_bias:@)
batch_normalization_4_gamma:@(
batch_normalization_4_beta:@/
!batch_normalization_4_moving_mean:@3
%batch_normalization_4_moving_variance:@!
dense_kernel:???

dense_bias:	?!
dense_1_kernel:	?@
dense_1_bias:@ 
dense_2_kernel:@
dense_2_bias:
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_biasbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_varianceconv2d_1_kernelconv2d_1_biasbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_2_kernelconv2d_2_biasbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_varianceconv2d_3_kernelconv2d_3_biasbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_4_kernelconv2d_4_biasbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancedense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_39832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_3_layer_call_fn_5198

inputs)
batch_normalization_3_gamma:@(
batch_normalization_3_beta:@/
!batch_normalization_3_moving_mean:@3
%batch_normalization_3_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!!@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_35492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????!!@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????!!@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????!!@
 
_user_specified_nameinputs
?
?
'__inference_conv2d_4_layer_call_fn_5216

inputs)
conv2d_4_kernel:@@
conv2d_4_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_kernelconv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_31462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*2
_input_shapes!
:?????????!!@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????!!@
 
_user_specified_nameinputs
?g
?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_3983

inputs.
conv2d_conv2d_kernel: 
conv2d_conv2d_bias:;
-batch_normalization_batch_normalization_gamma::
,batch_normalization_batch_normalization_beta:A
3batch_normalization_batch_normalization_moving_mean:E
7batch_normalization_batch_normalization_moving_variance:2
conv2d_1_conv2d_1_kernel: $
conv2d_1_conv2d_1_bias: ?
1batch_normalization_1_batch_normalization_1_gamma: >
0batch_normalization_1_batch_normalization_1_beta: E
7batch_normalization_1_batch_normalization_1_moving_mean: I
;batch_normalization_1_batch_normalization_1_moving_variance: 2
conv2d_2_conv2d_2_kernel: 0$
conv2d_2_conv2d_2_bias:0?
1batch_normalization_2_batch_normalization_2_gamma:0>
0batch_normalization_2_batch_normalization_2_beta:0E
7batch_normalization_2_batch_normalization_2_moving_mean:0I
;batch_normalization_2_batch_normalization_2_moving_variance:02
conv2d_3_conv2d_3_kernel:0@$
conv2d_3_conv2d_3_bias:@?
1batch_normalization_3_batch_normalization_3_gamma:@>
0batch_normalization_3_batch_normalization_3_beta:@E
7batch_normalization_3_batch_normalization_3_moving_mean:@I
;batch_normalization_3_batch_normalization_3_moving_variance:@2
conv2d_4_conv2d_4_kernel:@@$
conv2d_4_conv2d_4_bias:@?
1batch_normalization_4_batch_normalization_4_gamma:@>
0batch_normalization_4_batch_normalization_4_beta:@E
7batch_normalization_4_batch_normalization_4_moving_mean:@I
;batch_normalization_4_batch_normalization_4_moving_variance:@'
dense_dense_kernel:???
dense_dense_bias:	?)
dense_1_dense_1_kernel:	?@"
dense_1_dense_1_bias:@(
dense_2_dense_2_kernel:@"
dense_2_dense_2_bias:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_29942 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta3batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_38222-
+batch_normalization/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_30322"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:01batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta7batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37312/
-batch_normalization_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????##0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_30702"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:01batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta7batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????##0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_36402/
-batch_normalization_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_conv2d_3_kernelconv2d_3_conv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!!@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_31082"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:01batch_normalization_3_batch_normalization_3_gamma0batch_normalization_3_batch_normalization_3_beta7batch_normalization_3_batch_normalization_3_moving_mean;batch_normalization_3_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!!@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_35492/
-batch_normalization_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_conv2d_4_kernelconv2d_4_conv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_31462"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:01batch_normalization_4_batch_normalization_4_gamma0batch_normalization_4_batch_normalization_4_beta7batch_normalization_4_batch_normalization_4_moving_mean;batch_normalization_4_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34582/
-batch_normalization_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_31792
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_31922
dense/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_33782!
dropout/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_32142!
dense_1/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_33192#
!dropout_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_32362!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_5083

inputs?
%conv2d_readvariableop_conv2d_3_kernel:0@2
$biasadd_readvariableop_conv2d_3_bias:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_3_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????!!@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????!!@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*2
_input_shapes!
:?????????##0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????##0
 
_user_specified_nameinputs
?	
?
2__inference_batch_normalization_layer_call_fn_4820

inputs'
batch_normalization_gamma:&
batch_normalization_beta:-
batch_normalization_moving_mean:1
#batch_normalization_moving_variance:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_38222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*8
_input_shapes'
%:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_1_layer_call_fn_5420

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_32232
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3549

inputs8
*readvariableop_batch_normalization_3_gamma:@9
+readvariableop_1_batch_normalization_3_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????!!@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????!!@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????!!@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????!!@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4910

inputs8
*readvariableop_batch_normalization_1_gamma: 9
+readvariableop_1_batch_normalization_1_beta: O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean: U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????HH : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????HH 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????HH : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????HH 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5126

inputs8
*readvariableop_batch_normalization_3_gamma:@9
+readvariableop_1_batch_normalization_3_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_3179

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@?  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4957

inputs?
%conv2d_readvariableop_conv2d_2_kernel: 02
$biasadd_readvariableop_conv2d_2_bias:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##0*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_2_bias*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????##02
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????##02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*2
_input_shapes!
:?????????HH : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????HH 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_3032

inputs?
%conv2d_readvariableop_conv2d_1_kernel: 2
$biasadd_readvariableop_conv2d_1_bias: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_1_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????HH 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????HH 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4831

inputs?
%conv2d_readvariableop_conv2d_1_kernel: 2
$biasadd_readvariableop_conv2d_1_bias: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_1_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????HH 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????HH 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_conv2d_2_layer_call_fn_4964

inputs)
conv2d_2_kernel: 0
conv2d_2_bias:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_kernelconv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????##0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_30702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????##02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*2
_input_shapes!
:?????????HH : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????HH 
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_3201

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_5391

inputs7
$matmul_readvariableop_dense_1_kernel:	?@1
#biasadd_readvariableop_dense_1_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
4__inference_batch_normalization_4_layer_call_fn_5306

inputs)
batch_normalization_4_gamma:@(
batch_normalization_4_beta:@/
!batch_normalization_4_moving_mean:@3
%batch_normalization_4_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_28882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
2__inference_batch_normalization_layer_call_fn_4802

inputs'
batch_normalization_gamma:&
batch_normalization_beta:-
batch_normalization_moving_mean:1
#batch_normalization_moving_variance:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_23042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3167

inputs8
*readvariableop_batch_normalization_4_gamma:@9
+readvariableop_1_batch_normalization_4_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5144

inputs8
*readvariableop_batch_normalization_3_gamma:@9
+readvariableop_1_batch_normalization_3_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????!!@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????!!@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????!!@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????!!@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5288

inputs8
*readvariableop_batch_normalization_4_gamma:@9
+readvariableop_1_batch_normalization_4_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?#
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4464

inputsD
*conv2d_conv2d_readvariableop_conv2d_kernel:7
)conv2d_biasadd_readvariableop_conv2d_bias:J
<batch_normalization_readvariableop_batch_normalization_gamma:K
=batch_normalization_readvariableop_1_batch_normalization_beta:a
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:g
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:H
.conv2d_1_conv2d_readvariableop_conv2d_1_kernel: ;
-conv2d_1_biasadd_readvariableop_conv2d_1_bias: N
@batch_normalization_1_readvariableop_batch_normalization_1_gamma: O
Abatch_normalization_1_readvariableop_1_batch_normalization_1_beta: e
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean: k
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance: H
.conv2d_2_conv2d_readvariableop_conv2d_2_kernel: 0;
-conv2d_2_biasadd_readvariableop_conv2d_2_bias:0N
@batch_normalization_2_readvariableop_batch_normalization_2_gamma:0O
Abatch_normalization_2_readvariableop_1_batch_normalization_2_beta:0e
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:0k
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:0H
.conv2d_3_conv2d_readvariableop_conv2d_3_kernel:0@;
-conv2d_3_biasadd_readvariableop_conv2d_3_bias:@N
@batch_normalization_3_readvariableop_batch_normalization_3_gamma:@O
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta:@e
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:@k
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:@H
.conv2d_4_conv2d_readvariableop_conv2d_4_kernel:@@;
-conv2d_4_biasadd_readvariableop_conv2d_4_bias:@N
@batch_normalization_4_readvariableop_batch_normalization_4_gamma:@O
Abatch_normalization_4_readvariableop_1_batch_normalization_4_beta:@e
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:@k
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:@=
(dense_matmul_readvariableop_dense_kernel:???6
'dense_biasadd_readvariableop_dense_bias:	??
,dense_1_matmul_readvariableop_dense_1_kernel:	?@9
+dense_1_biasadd_readvariableop_dense_1_bias:@>
,dense_2_matmul_readvariableop_dense_2_kernel:@9
+dense_2_biasadd_readvariableop_dense_2_bias:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d/Relu?
"batch_normalization/ReadVariableOpReadVariableOp<batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp=batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH *
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????HH 2
conv2d_1/Relu?
$batch_normalization_1/ReadVariableOpReadVariableOp@batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOpAbatch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????HH : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##0*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp-conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
:0*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##02
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????##02
conv2d_2/Relu?
$batch_normalization_2/ReadVariableOpReadVariableOp@batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
:0*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOpAbatch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
:0*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????##0:0:0:0:0:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
:0@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp-conv2d_3_biasadd_readvariableop_conv2d_3_bias*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!!@2
conv2d_3/Relu?
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????!!@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp-conv2d_4_biasadd_readvariableop_conv2d_4_bias*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/Relu?
$batch_normalization_4/ReadVariableOpReadVariableOp@batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOpAbatch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@?  2
flatten/Const?
flatten/ReshapeReshape*batch_normalization_4/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*!
_output_shapes
:???*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu}
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel*
_output_shapes
:	?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_1/Relu?
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout_1/Identity?
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes

:@*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Sigmoidn
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_dense_2_layer_call_and_return_conditional_losses_3236

inputs6
$matmul_readvariableop_dense_2_kernel:@1
#biasadd_readvariableop_dense_2_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5234

inputs8
*readvariableop_batch_normalization_4_gamma:@9
+readvariableop_1_batch_normalization_4_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2304

inputs6
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_3_layer_call_fn_5189

inputs)
batch_normalization_3_gamma:@(
batch_normalization_3_beta:@/
!batch_normalization_3_moving_mean:@3
%batch_normalization_3_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!!@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_31292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????!!@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????!!@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????!!@
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3822

inputs6
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3091

inputs8
*readvariableop_batch_normalization_2_gamma:09
+readvariableop_1_batch_normalization_2_beta:0O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:0U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????##0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????##02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????##0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????##0
 
_user_specified_nameinputs
?g
?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4287
input_1.
conv2d_conv2d_kernel: 
conv2d_conv2d_bias:;
-batch_normalization_batch_normalization_gamma::
,batch_normalization_batch_normalization_beta:A
3batch_normalization_batch_normalization_moving_mean:E
7batch_normalization_batch_normalization_moving_variance:2
conv2d_1_conv2d_1_kernel: $
conv2d_1_conv2d_1_bias: ?
1batch_normalization_1_batch_normalization_1_gamma: >
0batch_normalization_1_batch_normalization_1_beta: E
7batch_normalization_1_batch_normalization_1_moving_mean: I
;batch_normalization_1_batch_normalization_1_moving_variance: 2
conv2d_2_conv2d_2_kernel: 0$
conv2d_2_conv2d_2_bias:0?
1batch_normalization_2_batch_normalization_2_gamma:0>
0batch_normalization_2_batch_normalization_2_beta:0E
7batch_normalization_2_batch_normalization_2_moving_mean:0I
;batch_normalization_2_batch_normalization_2_moving_variance:02
conv2d_3_conv2d_3_kernel:0@$
conv2d_3_conv2d_3_bias:@?
1batch_normalization_3_batch_normalization_3_gamma:@>
0batch_normalization_3_batch_normalization_3_beta:@E
7batch_normalization_3_batch_normalization_3_moving_mean:@I
;batch_normalization_3_batch_normalization_3_moving_variance:@2
conv2d_4_conv2d_4_kernel:@@$
conv2d_4_conv2d_4_bias:@?
1batch_normalization_4_batch_normalization_4_gamma:@>
0batch_normalization_4_batch_normalization_4_beta:@E
7batch_normalization_4_batch_normalization_4_moving_mean:@I
;batch_normalization_4_batch_normalization_4_moving_variance:@'
dense_dense_kernel:???
dense_dense_bias:	?)
dense_1_dense_1_kernel:	?@"
dense_1_dense_1_bias:@(
dense_2_dense_2_kernel:@"
dense_2_dense_2_bias:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_29942 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta3batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_38222-
+batch_normalization/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_30322"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:01batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta7batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37312/
-batch_normalization_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????##0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_30702"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:01batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta7batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????##0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_36402/
-batch_normalization_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_conv2d_3_kernelconv2d_3_conv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!!@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_31082"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:01batch_normalization_3_batch_normalization_3_gamma0batch_normalization_3_batch_normalization_3_beta7batch_normalization_3_batch_normalization_3_moving_mean;batch_normalization_3_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!!@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_35492/
-batch_normalization_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_conv2d_4_kernelconv2d_4_conv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_31462"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:01batch_normalization_4_batch_normalization_4_gamma0batch_normalization_4_batch_normalization_4_beta7batch_normalization_4_batch_normalization_4_moving_mean;batch_normalization_4_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34582/
-batch_normalization_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_31792
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_31922
dense/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_33782!
dropout/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_32142!
dense_1/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_33192#
!dropout_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_32362!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
a
(__inference_dropout_1_layer_call_fn_5425

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_33192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
%__inference_conv2d_layer_call_fn_4712

inputs'
conv2d_kernel:
conv2d_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_29942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
2__inference_batch_normalization_layer_call_fn_4811

inputs'
batch_normalization_gamma:&
batch_normalization_beta:-
batch_normalization_moving_mean:1
#batch_normalization_moving_variance:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_30152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*8
_input_shapes'
%:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2596

inputs8
*readvariableop_batch_normalization_2_gamma:09
+readvariableop_1_batch_normalization_2_beta:0O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:0U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?d
?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4231
input_1.
conv2d_conv2d_kernel: 
conv2d_conv2d_bias:;
-batch_normalization_batch_normalization_gamma::
,batch_normalization_batch_normalization_beta:A
3batch_normalization_batch_normalization_moving_mean:E
7batch_normalization_batch_normalization_moving_variance:2
conv2d_1_conv2d_1_kernel: $
conv2d_1_conv2d_1_bias: ?
1batch_normalization_1_batch_normalization_1_gamma: >
0batch_normalization_1_batch_normalization_1_beta: E
7batch_normalization_1_batch_normalization_1_moving_mean: I
;batch_normalization_1_batch_normalization_1_moving_variance: 2
conv2d_2_conv2d_2_kernel: 0$
conv2d_2_conv2d_2_bias:0?
1batch_normalization_2_batch_normalization_2_gamma:0>
0batch_normalization_2_batch_normalization_2_beta:0E
7batch_normalization_2_batch_normalization_2_moving_mean:0I
;batch_normalization_2_batch_normalization_2_moving_variance:02
conv2d_3_conv2d_3_kernel:0@$
conv2d_3_conv2d_3_bias:@?
1batch_normalization_3_batch_normalization_3_gamma:@>
0batch_normalization_3_batch_normalization_3_beta:@E
7batch_normalization_3_batch_normalization_3_moving_mean:@I
;batch_normalization_3_batch_normalization_3_moving_variance:@2
conv2d_4_conv2d_4_kernel:@@$
conv2d_4_conv2d_4_bias:@?
1batch_normalization_4_batch_normalization_4_gamma:@>
0batch_normalization_4_batch_normalization_4_beta:@E
7batch_normalization_4_batch_normalization_4_moving_mean:@I
;batch_normalization_4_batch_normalization_4_moving_variance:@'
dense_dense_kernel:???
dense_dense_bias:	?)
dense_1_dense_1_kernel:	?@"
dense_1_dense_1_bias:@(
dense_2_dense_2_kernel:@"
dense_2_dense_2_bias:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_29942 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta3batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_30152-
+batch_normalization/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_30322"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:01batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta7batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_30532/
-batch_normalization_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????##0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_30702"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:01batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta7batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????##0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_30912/
-batch_normalization_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_conv2d_3_kernelconv2d_3_conv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!!@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_31082"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:01batch_normalization_3_batch_normalization_3_gamma0batch_normalization_3_batch_normalization_3_beta7batch_normalization_3_batch_normalization_3_moving_mean;batch_normalization_3_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!!@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_31292/
-batch_normalization_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_conv2d_4_kernelconv2d_4_conv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_31462"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:01batch_normalization_4_batch_normalization_4_gamma0batch_normalization_4_batch_normalization_4_beta7batch_normalization_4_batch_normalization_4_moving_mean;batch_normalization_4_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_31672/
-batch_normalization_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_31792
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_31922
dense/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_32012
dropout/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_32142!
dense_1/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_32232
dropout_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_32362!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3129

inputs8
*readvariableop_batch_normalization_3_gamma:@9
+readvariableop_1_batch_normalization_3_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????!!@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????!!@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????!!@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????!!@
 
_user_specified_nameinputs
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_4705

inputs=
#conv2d_readvariableop_conv2d_kernel:0
"biasadd_readvariableop_conv2d_bias:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_conv2d_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_4_layer_call_fn_5315

inputs)
batch_normalization_4_gamma:@(
batch_normalization_4_beta:@/
!batch_normalization_4_moving_mean:@3
%batch_normalization_4_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_31672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_5358

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_4330
input_1'
conv2d_kernel:
conv2d_bias:'
batch_normalization_gamma:&
batch_normalization_beta:-
batch_normalization_moving_mean:1
#batch_normalization_moving_variance:)
conv2d_1_kernel: 
conv2d_1_bias: )
batch_normalization_1_gamma: (
batch_normalization_1_beta: /
!batch_normalization_1_moving_mean: 3
%batch_normalization_1_moving_variance: )
conv2d_2_kernel: 0
conv2d_2_bias:0)
batch_normalization_2_gamma:0(
batch_normalization_2_beta:0/
!batch_normalization_2_moving_mean:03
%batch_normalization_2_moving_variance:0)
conv2d_3_kernel:0@
conv2d_3_bias:@)
batch_normalization_3_gamma:@(
batch_normalization_3_beta:@/
!batch_normalization_3_moving_mean:@3
%batch_normalization_3_moving_variance:@)
conv2d_4_kernel:@@
conv2d_4_bias:@)
batch_normalization_4_gamma:@(
batch_normalization_4_beta:@/
!batch_normalization_4_moving_mean:@3
%batch_normalization_4_moving_variance:@!
dense_kernel:???

dense_bias:	?!
dense_1_kernel:	?@
dense_1_bias:@ 
dense_2_kernel:@
dense_2_bias:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelconv2d_biasbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_varianceconv2d_1_kernelconv2d_1_biasbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_2_kernelconv2d_2_biasbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_varianceconv2d_3_kernelconv2d_3_biasbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_4_kernelconv2d_4_biasbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancedense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_22462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
8__inference_single_image_simple_model_layer_call_fn_4653

inputs'
conv2d_kernel:
conv2d_bias:'
batch_normalization_gamma:&
batch_normalization_beta:-
batch_normalization_moving_mean:1
#batch_normalization_moving_variance:)
conv2d_1_kernel: 
conv2d_1_bias: )
batch_normalization_1_gamma: (
batch_normalization_1_beta: /
!batch_normalization_1_moving_mean: 3
%batch_normalization_1_moving_variance: )
conv2d_2_kernel: 0
conv2d_2_bias:0)
batch_normalization_2_gamma:0(
batch_normalization_2_beta:0/
!batch_normalization_2_moving_mean:03
%batch_normalization_2_moving_variance:0)
conv2d_3_kernel:0@
conv2d_3_bias:@)
batch_normalization_3_gamma:@(
batch_normalization_3_beta:@/
!batch_normalization_3_moving_mean:@3
%batch_normalization_3_moving_variance:@)
conv2d_4_kernel:@@
conv2d_4_bias:@)
batch_normalization_4_gamma:@(
batch_normalization_4_beta:@/
!batch_normalization_4_moving_mean:@3
%batch_normalization_4_moving_variance:@!
dense_kernel:???

dense_bias:	?!
dense_1_kernel:	?@
dense_1_bias:@ 
dense_2_kernel:@
dense_2_bias:
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_biasbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_varianceconv2d_1_kernelconv2d_1_biasbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_2_kernelconv2d_2_biasbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_varianceconv2d_3_kernelconv2d_3_biasbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_4_kernelconv2d_4_biasbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancedense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_32412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_dense_1_layer_call_fn_5398

inputs!
dense_1_kernel:	?@
dense_1_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_kerneldense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_32142
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2414

inputs8
*readvariableop_batch_normalization_1_gamma: 9
+readvariableop_1_batch_normalization_1_beta: O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean: U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4730

inputs6
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_3108

inputs?
%conv2d_readvariableop_conv2d_3_kernel:0@2
$biasadd_readvariableop_conv2d_3_bias:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_3_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????!!@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????!!@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????##0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????##0
 
_user_specified_nameinputs
?

?
A__inference_dense_2_layer_call_and_return_conditional_losses_5436

inputs6
$matmul_readvariableop_dense_2_kernel:@1
#biasadd_readvariableop_dense_2_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_5403

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2852

inputs8
*readvariableop_batch_normalization_4_gamma:@9
+readvariableop_1_batch_normalization_4_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3458

inputs8
*readvariableop_batch_normalization_4_gamma:@9
+readvariableop_1_batch_normalization_4_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3015

inputs6
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_1_layer_call_fn_4937

inputs)
batch_normalization_1_gamma: (
batch_normalization_1_beta: /
!batch_normalization_1_moving_mean: 3
%batch_normalization_1_moving_variance: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_30532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????HH 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????HH : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????HH 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3640

inputs8
*readvariableop_batch_normalization_2_gamma:09
+readvariableop_1_batch_normalization_2_beta:0O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:0U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????##0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????##02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????##0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????##0
 
_user_specified_nameinputs
?
?
'__inference_conv2d_3_layer_call_fn_5090

inputs)
conv2d_3_kernel:0@
conv2d_3_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_kernelconv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!!@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_31082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????!!@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*2
_input_shapes!
:?????????##0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????##0
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2450

inputs8
*readvariableop_batch_normalization_1_gamma: 9
+readvariableop_1_batch_normalization_1_beta: O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean: U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
4__inference_batch_normalization_3_layer_call_fn_5180

inputs)
batch_normalization_3_gamma:@(
batch_normalization_3_beta:@/
!batch_normalization_3_moving_mean:@3
%batch_normalization_3_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3053

inputs8
*readvariableop_batch_normalization_1_gamma: 9
+readvariableop_1_batch_normalization_1_beta: O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean: U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????HH : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????HH 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????HH : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????HH 
 
_user_specified_nameinputs
?
?
8__inference_single_image_simple_model_layer_call_fn_3280
input_1'
conv2d_kernel:
conv2d_bias:'
batch_normalization_gamma:&
batch_normalization_beta:-
batch_normalization_moving_mean:1
#batch_normalization_moving_variance:)
conv2d_1_kernel: 
conv2d_1_bias: )
batch_normalization_1_gamma: (
batch_normalization_1_beta: /
!batch_normalization_1_moving_mean: 3
%batch_normalization_1_moving_variance: )
conv2d_2_kernel: 0
conv2d_2_bias:0)
batch_normalization_2_gamma:0(
batch_normalization_2_beta:0/
!batch_normalization_2_moving_mean:03
%batch_normalization_2_moving_variance:0)
conv2d_3_kernel:0@
conv2d_3_bias:@)
batch_normalization_3_gamma:@(
batch_normalization_3_beta:@/
!batch_normalization_3_moving_mean:@3
%batch_normalization_3_moving_variance:@)
conv2d_4_kernel:@@
conv2d_4_bias:@)
batch_normalization_4_gamma:@(
batch_normalization_4_beta:@/
!batch_normalization_4_moving_mean:@3
%batch_normalization_4_moving_variance:@!
dense_kernel:???

dense_bias:	?!
dense_1_kernel:	?@
dense_1_bias:@ 
dense_2_kernel:@
dense_2_bias:
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelconv2d_biasbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_varianceconv2d_1_kernelconv2d_1_biasbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_2_kernelconv2d_2_biasbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_varianceconv2d_3_kernelconv2d_3_biasbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_4_kernelconv2d_4_biasbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancedense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_32412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?d
?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_3241

inputs.
conv2d_conv2d_kernel: 
conv2d_conv2d_bias:;
-batch_normalization_batch_normalization_gamma::
,batch_normalization_batch_normalization_beta:A
3batch_normalization_batch_normalization_moving_mean:E
7batch_normalization_batch_normalization_moving_variance:2
conv2d_1_conv2d_1_kernel: $
conv2d_1_conv2d_1_bias: ?
1batch_normalization_1_batch_normalization_1_gamma: >
0batch_normalization_1_batch_normalization_1_beta: E
7batch_normalization_1_batch_normalization_1_moving_mean: I
;batch_normalization_1_batch_normalization_1_moving_variance: 2
conv2d_2_conv2d_2_kernel: 0$
conv2d_2_conv2d_2_bias:0?
1batch_normalization_2_batch_normalization_2_gamma:0>
0batch_normalization_2_batch_normalization_2_beta:0E
7batch_normalization_2_batch_normalization_2_moving_mean:0I
;batch_normalization_2_batch_normalization_2_moving_variance:02
conv2d_3_conv2d_3_kernel:0@$
conv2d_3_conv2d_3_bias:@?
1batch_normalization_3_batch_normalization_3_gamma:@>
0batch_normalization_3_batch_normalization_3_beta:@E
7batch_normalization_3_batch_normalization_3_moving_mean:@I
;batch_normalization_3_batch_normalization_3_moving_variance:@2
conv2d_4_conv2d_4_kernel:@@$
conv2d_4_conv2d_4_bias:@?
1batch_normalization_4_batch_normalization_4_gamma:@>
0batch_normalization_4_batch_normalization_4_beta:@E
7batch_normalization_4_batch_normalization_4_moving_mean:@I
;batch_normalization_4_batch_normalization_4_moving_variance:@'
dense_dense_kernel:???
dense_dense_bias:	?)
dense_1_dense_1_kernel:	?@"
dense_1_dense_1_bias:@(
dense_2_dense_2_kernel:@"
dense_2_dense_2_bias:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_29942 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta3batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_30152-
+batch_normalization/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_30322"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:01batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta7batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_30532/
-batch_normalization_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????##0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_30702"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:01batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta7batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????##0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_30912/
-batch_normalization_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_conv2d_3_kernelconv2d_3_conv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!!@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_31082"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:01batch_normalization_3_batch_normalization_3_gamma0batch_normalization_3_batch_normalization_3_beta7batch_normalization_3_batch_normalization_3_moving_mean;batch_normalization_3_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!!@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_31292/
-batch_normalization_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_conv2d_4_kernelconv2d_4_conv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_31462"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:01batch_normalization_4_batch_normalization_4_gamma0batch_normalization_4_batch_normalization_4_beta7batch_normalization_4_batch_normalization_4_moving_mean;batch_normalization_4_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_31672/
-batch_normalization_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_31792
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_31922
dense/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_32012
dropout/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_32142!
dense_1/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_32232
dropout_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_32362!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5252

inputs8
*readvariableop_batch_normalization_4_gamma:@9
+readvariableop_1_batch_normalization_4_beta:@O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:@U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
_
&__inference_dropout_layer_call_fn_5380

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_33782
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_dense_2_layer_call_fn_5443

inputs 
dense_2_kernel:@
dense_2_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_32362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_2_layer_call_fn_5072

inputs)
batch_normalization_2_gamma:0(
batch_normalization_2_beta:0/
!batch_normalization_2_moving_mean:03
%batch_normalization_2_moving_variance:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????##0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_36402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????##02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:?????????##0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????##0
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4856

inputs8
*readvariableop_batch_normalization_1_gamma: 9
+readvariableop_1_batch_normalization_1_beta: O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean: U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????;
dense_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer-11
layer_with_weights-10
layer-12
layer-13
layer_with_weights-11
layer-14
layer-15
layer_with_weights-12
layer-16
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
axis
	gamma
 beta
!moving_mean
"moving_variance
#trainable_variables
$	variables
%regularization_losses
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2trainable_variables
3	variables
4regularization_losses
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

6kernel
7bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
<axis
	=gamma
>beta
?moving_mean
@moving_variance
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Ekernel
Fbias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Tkernel
Ubias
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_trainable_variables
`	variables
aregularization_losses
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

gkernel
hbias
itrainable_variables
j	variables
kregularization_losses
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

qkernel
rbias
strainable_variables
t	variables
uregularization_losses
v	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
wtrainable_variables
x	variables
yregularization_losses
z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?m? m?'m?(m?.m?/m?6m?7m?=m?>m?Em?Fm?Lm?Mm?Tm?Um?[m?\m?gm?hm?qm?rm?{m?|m?v?v?v? v?'v?(v?.v?/v?6v?7v?=v?>v?Ev?Fv?Lv?Mv?Tv?Uv?[v?\v?gv?hv?qv?rv?{v?|v?"
	optimizer
?
0
1
2
 3
'4
(5
.6
/7
68
79
=10
>11
E12
F13
L14
M15
T16
U17
[18
\19
g20
h21
q22
r23
{24
|25"
trackable_list_wrapper
?
0
1
2
 3
!4
"5
'6
(7
.8
/9
010
111
612
713
=14
>15
?16
@17
E18
F19
L20
M21
N22
O23
T24
U25
[26
\27
]28
^29
g30
h31
q32
r33
{34
|35"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
trainable_variables
?layer_metrics
	variables
?non_trainable_variables
 ?layer_regularization_losses
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
trainable_variables
?layer_metrics
	variables
?non_trainable_variables
 ?layer_regularization_losses
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
.
0
 1"
trackable_list_wrapper
<
0
 1
!2
"3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
#trainable_variables
?layer_metrics
$	variables
?non_trainable_variables
 ?layer_regularization_losses
%regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_1/kernel
: 2conv2d_1/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
)trainable_variables
?layer_metrics
*	variables
?non_trainable_variables
 ?layer_regularization_losses
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
.
.0
/1"
trackable_list_wrapper
<
.0
/1
02
13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
2trainable_variables
?layer_metrics
3	variables
?non_trainable_variables
 ?layer_regularization_losses
4regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 02conv2d_2/kernel
:02conv2d_2/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
8trainable_variables
?layer_metrics
9	variables
?non_trainable_variables
 ?layer_regularization_losses
:regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'02batch_normalization_2/gamma
(:&02batch_normalization_2/beta
1:/0 (2!batch_normalization_2/moving_mean
5:30 (2%batch_normalization_2/moving_variance
.
=0
>1"
trackable_list_wrapper
<
=0
>1
?2
@3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
Atrainable_variables
?layer_metrics
B	variables
?non_trainable_variables
 ?layer_regularization_losses
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'0@2conv2d_3/kernel
:@2conv2d_3/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
Gtrainable_variables
?layer_metrics
H	variables
?non_trainable_variables
 ?layer_regularization_losses
Iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
.
L0
M1"
trackable_list_wrapper
<
L0
M1
N2
O3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
Ptrainable_variables
?layer_metrics
Q	variables
?non_trainable_variables
 ?layer_regularization_losses
Rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_4/kernel
:@2conv2d_4/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
Vtrainable_variables
?layer_metrics
W	variables
?non_trainable_variables
 ?layer_regularization_losses
Xregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
.
[0
\1"
trackable_list_wrapper
<
[0
\1
]2
^3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
_trainable_variables
?layer_metrics
`	variables
?non_trainable_variables
 ?layer_regularization_losses
aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
ctrainable_variables
?layer_metrics
d	variables
?non_trainable_variables
 ?layer_regularization_losses
eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:???2dense/kernel
:?2
dense/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
itrainable_variables
?layer_metrics
j	variables
?non_trainable_variables
 ?layer_regularization_losses
kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
mtrainable_variables
?layer_metrics
n	variables
?non_trainable_variables
 ?layer_regularization_losses
oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?@2dense_1/kernel
:@2dense_1/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
strainable_variables
?layer_metrics
t	variables
?non_trainable_variables
 ?layer_regularization_losses
uregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
wtrainable_variables
?layer_metrics
x	variables
?non_trainable_variables
 ?layer_regularization_losses
yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_2/kernel
:2dense_2/bias
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
}trainable_variables
?layer_metrics
~	variables
?non_trainable_variables
 ?layer_regularization_losses
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
(
?0"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_dict_wrapper
f
!0
"1
02
13
?4
@5
N6
O7
]8
^9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
5:32training/Adam/conv2d/kernel/m
':%2training/Adam/conv2d/bias/m
5:32)training/Adam/batch_normalization/gamma/m
4:22(training/Adam/batch_normalization/beta/m
7:5 2training/Adam/conv2d_1/kernel/m
):' 2training/Adam/conv2d_1/bias/m
7:5 2+training/Adam/batch_normalization_1/gamma/m
6:4 2*training/Adam/batch_normalization_1/beta/m
7:5 02training/Adam/conv2d_2/kernel/m
):'02training/Adam/conv2d_2/bias/m
7:502+training/Adam/batch_normalization_2/gamma/m
6:402*training/Adam/batch_normalization_2/beta/m
7:50@2training/Adam/conv2d_3/kernel/m
):'@2training/Adam/conv2d_3/bias/m
7:5@2+training/Adam/batch_normalization_3/gamma/m
6:4@2*training/Adam/batch_normalization_3/beta/m
7:5@@2training/Adam/conv2d_4/kernel/m
):'@2training/Adam/conv2d_4/bias/m
7:5@2+training/Adam/batch_normalization_4/gamma/m
6:4@2*training/Adam/batch_normalization_4/beta/m
/:-???2training/Adam/dense/kernel/m
':%?2training/Adam/dense/bias/m
/:-	?@2training/Adam/dense_1/kernel/m
(:&@2training/Adam/dense_1/bias/m
.:,@2training/Adam/dense_2/kernel/m
(:&2training/Adam/dense_2/bias/m
5:32training/Adam/conv2d/kernel/v
':%2training/Adam/conv2d/bias/v
5:32)training/Adam/batch_normalization/gamma/v
4:22(training/Adam/batch_normalization/beta/v
7:5 2training/Adam/conv2d_1/kernel/v
):' 2training/Adam/conv2d_1/bias/v
7:5 2+training/Adam/batch_normalization_1/gamma/v
6:4 2*training/Adam/batch_normalization_1/beta/v
7:5 02training/Adam/conv2d_2/kernel/v
):'02training/Adam/conv2d_2/bias/v
7:502+training/Adam/batch_normalization_2/gamma/v
6:402*training/Adam/batch_normalization_2/beta/v
7:50@2training/Adam/conv2d_3/kernel/v
):'@2training/Adam/conv2d_3/bias/v
7:5@2+training/Adam/batch_normalization_3/gamma/v
6:4@2*training/Adam/batch_normalization_3/beta/v
7:5@@2training/Adam/conv2d_4/kernel/v
):'@2training/Adam/conv2d_4/bias/v
7:5@2+training/Adam/batch_normalization_4/gamma/v
6:4@2*training/Adam/batch_normalization_4/beta/v
/:-???2training/Adam/dense/kernel/v
':%?2training/Adam/dense/bias/v
/:-	?@2training/Adam/dense_1/kernel/v
(:&@2training/Adam/dense_1/bias/v
.:,@2training/Adam/dense_2/kernel/v
(:&2training/Adam/dense_2/bias/v
?2?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4464
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4612
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4231
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4287?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_2246input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_single_image_simple_model_layer_call_fn_3280
8__inference_single_image_simple_model_layer_call_fn_4653
8__inference_single_image_simple_model_layer_call_fn_4694
8__inference_single_image_simple_model_layer_call_fn_4175?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_conv2d_layer_call_and_return_conditional_losses_4705?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv2d_layer_call_fn_4712?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4730
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4748
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4766
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4784?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
2__inference_batch_normalization_layer_call_fn_4793
2__inference_batch_normalization_layer_call_fn_4802
2__inference_batch_normalization_layer_call_fn_4811
2__inference_batch_normalization_layer_call_fn_4820?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4831?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_1_layer_call_fn_4838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4856
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4874
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4892
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4910?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_1_layer_call_fn_4919
4__inference_batch_normalization_1_layer_call_fn_4928
4__inference_batch_normalization_1_layer_call_fn_4937
4__inference_batch_normalization_1_layer_call_fn_4946?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4957?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_2_layer_call_fn_4964?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4982
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5000
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5018
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5036?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_2_layer_call_fn_5045
4__inference_batch_normalization_2_layer_call_fn_5054
4__inference_batch_normalization_2_layer_call_fn_5063
4__inference_batch_normalization_2_layer_call_fn_5072?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_5083?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_3_layer_call_fn_5090?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5108
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5126
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5144
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5162?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_3_layer_call_fn_5171
4__inference_batch_normalization_3_layer_call_fn_5180
4__inference_batch_normalization_3_layer_call_fn_5189
4__inference_batch_normalization_3_layer_call_fn_5198?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5209?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_4_layer_call_fn_5216?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5234
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5252
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5270
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5288?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_4_layer_call_fn_5297
4__inference_batch_normalization_4_layer_call_fn_5306
4__inference_batch_normalization_4_layer_call_fn_5315
4__inference_batch_normalization_4_layer_call_fn_5324?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_flatten_layer_call_and_return_conditional_losses_5330?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_flatten_layer_call_fn_5335?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_5346?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_5353?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dropout_layer_call_and_return_conditional_losses_5358
A__inference_dropout_layer_call_and_return_conditional_losses_5370?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dropout_layer_call_fn_5375
&__inference_dropout_layer_call_fn_5380?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_5391?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_5398?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout_1_layer_call_and_return_conditional_losses_5403
C__inference_dropout_1_layer_call_and_return_conditional_losses_5415?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_1_layer_call_fn_5420
(__inference_dropout_1_layer_call_fn_5425?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dense_2_layer_call_and_return_conditional_losses_5436?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_2_layer_call_fn_5443?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_4330input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_2246?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|:?7
0?-
+?(
input_1???????????
? "1?.
,
dense_2!?
dense_2??????????
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4856?./01M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4874?./01M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4892r./01;?8
1?.
(?%
inputs?????????HH 
p 
? "-?*
#? 
0?????????HH 
? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4910r./01;?8
1?.
(?%
inputs?????????HH 
p
? "-?*
#? 
0?????????HH 
? ?
4__inference_batch_normalization_1_layer_call_fn_4919?./01M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
4__inference_batch_normalization_1_layer_call_fn_4928?./01M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
4__inference_batch_normalization_1_layer_call_fn_4937e./01;?8
1?.
(?%
inputs?????????HH 
p 
? " ??????????HH ?
4__inference_batch_normalization_1_layer_call_fn_4946e./01;?8
1?.
(?%
inputs?????????HH 
p
? " ??????????HH ?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4982?=>?@M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5000?=>?@M?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5018r=>?@;?8
1?.
(?%
inputs?????????##0
p 
? "-?*
#? 
0?????????##0
? ?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5036r=>?@;?8
1?.
(?%
inputs?????????##0
p
? "-?*
#? 
0?????????##0
? ?
4__inference_batch_normalization_2_layer_call_fn_5045?=>?@M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
4__inference_batch_normalization_2_layer_call_fn_5054?=>?@M?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
4__inference_batch_normalization_2_layer_call_fn_5063e=>?@;?8
1?.
(?%
inputs?????????##0
p 
? " ??????????##0?
4__inference_batch_normalization_2_layer_call_fn_5072e=>?@;?8
1?.
(?%
inputs?????????##0
p
? " ??????????##0?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5108?LMNOM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5126?LMNOM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5144rLMNO;?8
1?.
(?%
inputs?????????!!@
p 
? "-?*
#? 
0?????????!!@
? ?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5162rLMNO;?8
1?.
(?%
inputs?????????!!@
p
? "-?*
#? 
0?????????!!@
? ?
4__inference_batch_normalization_3_layer_call_fn_5171?LMNOM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
4__inference_batch_normalization_3_layer_call_fn_5180?LMNOM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
4__inference_batch_normalization_3_layer_call_fn_5189eLMNO;?8
1?.
(?%
inputs?????????!!@
p 
? " ??????????!!@?
4__inference_batch_normalization_3_layer_call_fn_5198eLMNO;?8
1?.
(?%
inputs?????????!!@
p
? " ??????????!!@?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5234?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5252?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5270r[\]^;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5288r[\]^;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
4__inference_batch_normalization_4_layer_call_fn_5297?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
4__inference_batch_normalization_4_layer_call_fn_5306?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
4__inference_batch_normalization_4_layer_call_fn_5315e[\]^;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
4__inference_batch_normalization_4_layer_call_fn_5324e[\]^;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4730? !"M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4748? !"M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4766v !"=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4784v !"=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
2__inference_batch_normalization_layer_call_fn_4793? !"M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
2__inference_batch_normalization_layer_call_fn_4802? !"M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
2__inference_batch_normalization_layer_call_fn_4811i !"=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
2__inference_batch_normalization_layer_call_fn_4820i !"=?:
3?0
*?'
inputs???????????
p
? ""?????????????
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4831n'(9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????HH 
? ?
'__inference_conv2d_1_layer_call_fn_4838a'(9?6
/?,
*?'
inputs???????????
? " ??????????HH ?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4957l677?4
-?*
(?%
inputs?????????HH 
? "-?*
#? 
0?????????##0
? ?
'__inference_conv2d_2_layer_call_fn_4964_677?4
-?*
(?%
inputs?????????HH 
? " ??????????##0?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_5083lEF7?4
-?*
(?%
inputs?????????##0
? "-?*
#? 
0?????????!!@
? ?
'__inference_conv2d_3_layer_call_fn_5090_EF7?4
-?*
(?%
inputs?????????##0
? " ??????????!!@?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5209lTU7?4
-?*
(?%
inputs?????????!!@
? "-?*
#? 
0?????????@
? ?
'__inference_conv2d_4_layer_call_fn_5216_TU7?4
-?*
(?%
inputs?????????!!@
? " ??????????@?
@__inference_conv2d_layer_call_and_return_conditional_losses_4705p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
%__inference_conv2d_layer_call_fn_4712c9?6
/?,
*?'
inputs???????????
? ""?????????????
A__inference_dense_1_layer_call_and_return_conditional_losses_5391]qr0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? z
&__inference_dense_1_layer_call_fn_5398Pqr0?-
&?#
!?
inputs??????????
? "??????????@?
A__inference_dense_2_layer_call_and_return_conditional_losses_5436\{|/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? y
&__inference_dense_2_layer_call_fn_5443O{|/?,
%?"
 ?
inputs?????????@
? "???????????
?__inference_dense_layer_call_and_return_conditional_losses_5346_gh1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? z
$__inference_dense_layer_call_fn_5353Rgh1?.
'?$
"?
inputs???????????
? "????????????
C__inference_dropout_1_layer_call_and_return_conditional_losses_5403\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_5415\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? {
(__inference_dropout_1_layer_call_fn_5420O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@{
(__inference_dropout_1_layer_call_fn_5425O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
A__inference_dropout_layer_call_and_return_conditional_losses_5358^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_5370^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? {
&__inference_dropout_layer_call_fn_5375Q4?1
*?'
!?
inputs??????????
p 
? "???????????{
&__inference_dropout_layer_call_fn_5380Q4?1
*?'
!?
inputs??????????
p
? "????????????
A__inference_flatten_layer_call_and_return_conditional_losses_5330b7?4
-?*
(?%
inputs?????????@
? "'?$
?
0???????????
? 
&__inference_flatten_layer_call_fn_5335U7?4
-?*
(?%
inputs?????????@
? "?????????????
"__inference_signature_wrapper_4330?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|E?B
? 
;?8
6
input_1+?(
input_1???????????"1?.
,
dense_2!?
dense_2??????????
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4231?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|B??
8?5
+?(
input_1???????????
p 

 
? "%?"
?
0?????????
? ?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4287?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|B??
8?5
+?(
input_1???????????
p

 
? "%?"
?
0?????????
? ?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4464?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_4612?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
8__inference_single_image_simple_model_layer_call_fn_3280?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|B??
8?5
+?(
input_1???????????
p 

 
? "???????????
8__inference_single_image_simple_model_layer_call_fn_4175?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|B??
8?5
+?(
input_1???????????
p

 
? "???????????
8__inference_single_image_simple_model_layer_call_fn_4653?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
8__inference_single_image_simple_model_layer_call_fn_4694?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|A?>
7?4
*?'
inputs???????????
p

 
? "??????????