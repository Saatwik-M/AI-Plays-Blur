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
?
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:*
dtype0
?
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_10/gamma
?
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_10/beta
?
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_10/moving_mean
?
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_10/moving_variance
?
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
: *
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
: *
dtype0
?
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_11/gamma
?
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_11/beta
?
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_11/moving_mean
?
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_11/moving_variance
?
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: 0*
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
:0*
dtype0
?
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*-
shared_namebatch_normalization_12/gamma
?
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:0*
dtype0
?
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_12/beta
?
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:0*
dtype0
?
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"batch_normalization_12/moving_mean
?
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:0*
dtype0
?
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*7
shared_name(&batch_normalization_12/moving_variance
?
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:0*
dtype0
?
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:0@*
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_13/gamma
?
0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_13/beta
?
/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_13/moving_mean
?
6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_13/moving_variance
?
:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_14/gamma
?
0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_14/beta
?
/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_14/moving_mean
?
6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_14/moving_variance
?
:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes
:@*
dtype0
{
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_namedense_6/kernel
t
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*!
_output_shapes
:???*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:?*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	?@*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:@*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:@*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
|
training_2/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining_2/Adam/iter
u
(training_2/Adam/iter/Read/ReadVariableOpReadVariableOptraining_2/Adam/iter*
_output_shapes
: *
dtype0	
?
training_2/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_2/Adam/beta_1
y
*training_2/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_1*
_output_shapes
: *
dtype0
?
training_2/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_2/Adam/beta_2
y
*training_2/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_2*
_output_shapes
: *
dtype0
~
training_2/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining_2/Adam/decay
w
)training_2/Adam/decay/Read/ReadVariableOpReadVariableOptraining_2/Adam/decay*
_output_shapes
: *
dtype0
?
training_2/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining_2/Adam/learning_rate
?
1training_2/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_2/Adam/learning_rate*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
"training_2/Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_2/Adam/conv2d_10/kernel/m
?
6training_2/Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOp"training_2/Adam/conv2d_10/kernel/m*&
_output_shapes
:*
dtype0
?
 training_2/Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_2/Adam/conv2d_10/bias/m
?
4training_2/Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOp training_2/Adam/conv2d_10/bias/m*
_output_shapes
:*
dtype0
?
.training_2/Adam/batch_normalization_10/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.training_2/Adam/batch_normalization_10/gamma/m
?
Btraining_2/Adam/batch_normalization_10/gamma/m/Read/ReadVariableOpReadVariableOp.training_2/Adam/batch_normalization_10/gamma/m*
_output_shapes
:*
dtype0
?
-training_2/Adam/batch_normalization_10/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-training_2/Adam/batch_normalization_10/beta/m
?
Atraining_2/Adam/batch_normalization_10/beta/m/Read/ReadVariableOpReadVariableOp-training_2/Adam/batch_normalization_10/beta/m*
_output_shapes
:*
dtype0
?
"training_2/Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"training_2/Adam/conv2d_11/kernel/m
?
6training_2/Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOp"training_2/Adam/conv2d_11/kernel/m*&
_output_shapes
: *
dtype0
?
 training_2/Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" training_2/Adam/conv2d_11/bias/m
?
4training_2/Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOp training_2/Adam/conv2d_11/bias/m*
_output_shapes
: *
dtype0
?
.training_2/Adam/batch_normalization_11/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.training_2/Adam/batch_normalization_11/gamma/m
?
Btraining_2/Adam/batch_normalization_11/gamma/m/Read/ReadVariableOpReadVariableOp.training_2/Adam/batch_normalization_11/gamma/m*
_output_shapes
: *
dtype0
?
-training_2/Adam/batch_normalization_11/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-training_2/Adam/batch_normalization_11/beta/m
?
Atraining_2/Adam/batch_normalization_11/beta/m/Read/ReadVariableOpReadVariableOp-training_2/Adam/batch_normalization_11/beta/m*
_output_shapes
: *
dtype0
?
"training_2/Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*3
shared_name$"training_2/Adam/conv2d_12/kernel/m
?
6training_2/Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOp"training_2/Adam/conv2d_12/kernel/m*&
_output_shapes
: 0*
dtype0
?
 training_2/Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" training_2/Adam/conv2d_12/bias/m
?
4training_2/Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOp training_2/Adam/conv2d_12/bias/m*
_output_shapes
:0*
dtype0
?
.training_2/Adam/batch_normalization_12/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*?
shared_name0.training_2/Adam/batch_normalization_12/gamma/m
?
Btraining_2/Adam/batch_normalization_12/gamma/m/Read/ReadVariableOpReadVariableOp.training_2/Adam/batch_normalization_12/gamma/m*
_output_shapes
:0*
dtype0
?
-training_2/Adam/batch_normalization_12/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*>
shared_name/-training_2/Adam/batch_normalization_12/beta/m
?
Atraining_2/Adam/batch_normalization_12/beta/m/Read/ReadVariableOpReadVariableOp-training_2/Adam/batch_normalization_12/beta/m*
_output_shapes
:0*
dtype0
?
"training_2/Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*3
shared_name$"training_2/Adam/conv2d_13/kernel/m
?
6training_2/Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOp"training_2/Adam/conv2d_13/kernel/m*&
_output_shapes
:0@*
dtype0
?
 training_2/Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_2/Adam/conv2d_13/bias/m
?
4training_2/Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOp training_2/Adam/conv2d_13/bias/m*
_output_shapes
:@*
dtype0
?
.training_2/Adam/batch_normalization_13/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.training_2/Adam/batch_normalization_13/gamma/m
?
Btraining_2/Adam/batch_normalization_13/gamma/m/Read/ReadVariableOpReadVariableOp.training_2/Adam/batch_normalization_13/gamma/m*
_output_shapes
:@*
dtype0
?
-training_2/Adam/batch_normalization_13/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-training_2/Adam/batch_normalization_13/beta/m
?
Atraining_2/Adam/batch_normalization_13/beta/m/Read/ReadVariableOpReadVariableOp-training_2/Adam/batch_normalization_13/beta/m*
_output_shapes
:@*
dtype0
?
"training_2/Adam/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*3
shared_name$"training_2/Adam/conv2d_14/kernel/m
?
6training_2/Adam/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOp"training_2/Adam/conv2d_14/kernel/m*&
_output_shapes
:@@*
dtype0
?
 training_2/Adam/conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_2/Adam/conv2d_14/bias/m
?
4training_2/Adam/conv2d_14/bias/m/Read/ReadVariableOpReadVariableOp training_2/Adam/conv2d_14/bias/m*
_output_shapes
:@*
dtype0
?
.training_2/Adam/batch_normalization_14/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.training_2/Adam/batch_normalization_14/gamma/m
?
Btraining_2/Adam/batch_normalization_14/gamma/m/Read/ReadVariableOpReadVariableOp.training_2/Adam/batch_normalization_14/gamma/m*
_output_shapes
:@*
dtype0
?
-training_2/Adam/batch_normalization_14/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-training_2/Adam/batch_normalization_14/beta/m
?
Atraining_2/Adam/batch_normalization_14/beta/m/Read/ReadVariableOpReadVariableOp-training_2/Adam/batch_normalization_14/beta/m*
_output_shapes
:@*
dtype0
?
 training_2/Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*1
shared_name" training_2/Adam/dense_6/kernel/m
?
4training_2/Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_6/kernel/m*!
_output_shapes
:???*
dtype0
?
training_2/Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name training_2/Adam/dense_6/bias/m
?
2training_2/Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_6/bias/m*
_output_shapes	
:?*
dtype0
?
 training_2/Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*1
shared_name" training_2/Adam/dense_7/kernel/m
?
4training_2/Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_7/kernel/m*
_output_shapes
:	?@*
dtype0
?
training_2/Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name training_2/Adam/dense_7/bias/m
?
2training_2/Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_7/bias/m*
_output_shapes
:@*
dtype0
?
 training_2/Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*1
shared_name" training_2/Adam/dense_8/kernel/m
?
4training_2/Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_8/kernel/m*
_output_shapes

:@*
dtype0
?
training_2/Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training_2/Adam/dense_8/bias/m
?
2training_2/Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_8/bias/m*
_output_shapes
:*
dtype0
?
"training_2/Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_2/Adam/conv2d_10/kernel/v
?
6training_2/Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOp"training_2/Adam/conv2d_10/kernel/v*&
_output_shapes
:*
dtype0
?
 training_2/Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_2/Adam/conv2d_10/bias/v
?
4training_2/Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOp training_2/Adam/conv2d_10/bias/v*
_output_shapes
:*
dtype0
?
.training_2/Adam/batch_normalization_10/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.training_2/Adam/batch_normalization_10/gamma/v
?
Btraining_2/Adam/batch_normalization_10/gamma/v/Read/ReadVariableOpReadVariableOp.training_2/Adam/batch_normalization_10/gamma/v*
_output_shapes
:*
dtype0
?
-training_2/Adam/batch_normalization_10/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-training_2/Adam/batch_normalization_10/beta/v
?
Atraining_2/Adam/batch_normalization_10/beta/v/Read/ReadVariableOpReadVariableOp-training_2/Adam/batch_normalization_10/beta/v*
_output_shapes
:*
dtype0
?
"training_2/Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"training_2/Adam/conv2d_11/kernel/v
?
6training_2/Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOp"training_2/Adam/conv2d_11/kernel/v*&
_output_shapes
: *
dtype0
?
 training_2/Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" training_2/Adam/conv2d_11/bias/v
?
4training_2/Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOp training_2/Adam/conv2d_11/bias/v*
_output_shapes
: *
dtype0
?
.training_2/Adam/batch_normalization_11/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.training_2/Adam/batch_normalization_11/gamma/v
?
Btraining_2/Adam/batch_normalization_11/gamma/v/Read/ReadVariableOpReadVariableOp.training_2/Adam/batch_normalization_11/gamma/v*
_output_shapes
: *
dtype0
?
-training_2/Adam/batch_normalization_11/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-training_2/Adam/batch_normalization_11/beta/v
?
Atraining_2/Adam/batch_normalization_11/beta/v/Read/ReadVariableOpReadVariableOp-training_2/Adam/batch_normalization_11/beta/v*
_output_shapes
: *
dtype0
?
"training_2/Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*3
shared_name$"training_2/Adam/conv2d_12/kernel/v
?
6training_2/Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOp"training_2/Adam/conv2d_12/kernel/v*&
_output_shapes
: 0*
dtype0
?
 training_2/Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" training_2/Adam/conv2d_12/bias/v
?
4training_2/Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOp training_2/Adam/conv2d_12/bias/v*
_output_shapes
:0*
dtype0
?
.training_2/Adam/batch_normalization_12/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*?
shared_name0.training_2/Adam/batch_normalization_12/gamma/v
?
Btraining_2/Adam/batch_normalization_12/gamma/v/Read/ReadVariableOpReadVariableOp.training_2/Adam/batch_normalization_12/gamma/v*
_output_shapes
:0*
dtype0
?
-training_2/Adam/batch_normalization_12/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*>
shared_name/-training_2/Adam/batch_normalization_12/beta/v
?
Atraining_2/Adam/batch_normalization_12/beta/v/Read/ReadVariableOpReadVariableOp-training_2/Adam/batch_normalization_12/beta/v*
_output_shapes
:0*
dtype0
?
"training_2/Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*3
shared_name$"training_2/Adam/conv2d_13/kernel/v
?
6training_2/Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOp"training_2/Adam/conv2d_13/kernel/v*&
_output_shapes
:0@*
dtype0
?
 training_2/Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_2/Adam/conv2d_13/bias/v
?
4training_2/Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOp training_2/Adam/conv2d_13/bias/v*
_output_shapes
:@*
dtype0
?
.training_2/Adam/batch_normalization_13/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.training_2/Adam/batch_normalization_13/gamma/v
?
Btraining_2/Adam/batch_normalization_13/gamma/v/Read/ReadVariableOpReadVariableOp.training_2/Adam/batch_normalization_13/gamma/v*
_output_shapes
:@*
dtype0
?
-training_2/Adam/batch_normalization_13/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-training_2/Adam/batch_normalization_13/beta/v
?
Atraining_2/Adam/batch_normalization_13/beta/v/Read/ReadVariableOpReadVariableOp-training_2/Adam/batch_normalization_13/beta/v*
_output_shapes
:@*
dtype0
?
"training_2/Adam/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*3
shared_name$"training_2/Adam/conv2d_14/kernel/v
?
6training_2/Adam/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOp"training_2/Adam/conv2d_14/kernel/v*&
_output_shapes
:@@*
dtype0
?
 training_2/Adam/conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_2/Adam/conv2d_14/bias/v
?
4training_2/Adam/conv2d_14/bias/v/Read/ReadVariableOpReadVariableOp training_2/Adam/conv2d_14/bias/v*
_output_shapes
:@*
dtype0
?
.training_2/Adam/batch_normalization_14/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.training_2/Adam/batch_normalization_14/gamma/v
?
Btraining_2/Adam/batch_normalization_14/gamma/v/Read/ReadVariableOpReadVariableOp.training_2/Adam/batch_normalization_14/gamma/v*
_output_shapes
:@*
dtype0
?
-training_2/Adam/batch_normalization_14/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-training_2/Adam/batch_normalization_14/beta/v
?
Atraining_2/Adam/batch_normalization_14/beta/v/Read/ReadVariableOpReadVariableOp-training_2/Adam/batch_normalization_14/beta/v*
_output_shapes
:@*
dtype0
?
 training_2/Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*1
shared_name" training_2/Adam/dense_6/kernel/v
?
4training_2/Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_6/kernel/v*!
_output_shapes
:???*
dtype0
?
training_2/Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name training_2/Adam/dense_6/bias/v
?
2training_2/Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_6/bias/v*
_output_shapes	
:?*
dtype0
?
 training_2/Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*1
shared_name" training_2/Adam/dense_7/kernel/v
?
4training_2/Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_7/kernel/v*
_output_shapes
:	?@*
dtype0
?
training_2/Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name training_2/Adam/dense_7/bias/v
?
2training_2/Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_7/bias/v*
_output_shapes
:@*
dtype0
?
 training_2/Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*1
shared_name" training_2/Adam/dense_8/kernel/v
?
4training_2/Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_8/kernel/v*
_output_shapes

:@*
dtype0
?
training_2/Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training_2/Adam/dense_8/bias/v
?
2training_2/Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_8/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ܙ
valueљB͙ Bř
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?
axis
	gamma
 beta
!moving_mean
"moving_variance
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2	variables
3trainable_variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?
<axis
	=gamma
>beta
?moving_mean
@moving_variance
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
h

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
h

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
R
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
h

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
R
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
h

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
R
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
i

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?m? m?'m?(m?.m?/m?6m?7m?=m?>m?Em?Fm?Lm?Mm?Tm?Um?[m?\m?gm?hm?qm?rm?{m?|m?v?v?v? v?'v?(v?.v?/v?6v?7v?=v?>v?Ev?Fv?Lv?Mv?Tv?Uv?[v?\v?gv?hv?qv?rv?{v?|v?
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
 
?
?metrics
	variables
trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?layers
 
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
?metrics
	variables
trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
 1
!2
"3

0
 1
 
?
?metrics
#	variables
$trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
%regularization_losses
?layers
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
?metrics
)	variables
*trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
+regularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
02
13

.0
/1
 
?
?metrics
2	variables
3trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
4regularization_losses
?layers
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
?
?metrics
8	variables
9trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
:regularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
?2
@3

=0
>1
 
?
?metrics
A	variables
Btrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
Cregularization_losses
?layers
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1

E0
F1
 
?
?metrics
G	variables
Htrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
Iregularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
N2
O3

L0
M1
 
?
?metrics
P	variables
Qtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
Rregularization_losses
?layers
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

T0
U1
 
?
?metrics
V	variables
Wtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
Xregularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_14/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_14/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_14/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_14/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

[0
\1
]2
^3

[0
\1
 
?
?metrics
_	variables
`trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
aregularization_losses
?layers
 
 
 
?
?metrics
c	variables
dtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
eregularization_losses
?layers
[Y
VARIABLE_VALUEdense_6/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_6/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
?
?metrics
i	variables
jtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
kregularization_losses
?layers
 
 
 
?
?metrics
m	variables
ntrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
oregularization_losses
?layers
[Y
VARIABLE_VALUEdense_7/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_7/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1

q0
r1
 
?
?metrics
s	variables
ttrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
uregularization_losses
?layers
 
 
 
?
?metrics
w	variables
xtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
yregularization_losses
?layers
[Y
VARIABLE_VALUEdense_8/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_8/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

{0
|1
 
?
?metrics
}	variables
~trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?layers
SQ
VARIABLE_VALUEtraining_2/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_2/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_2/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining_2/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining_2/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

?0
 
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
QO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE"training_2/Adam/conv2d_10/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/conv2d_10/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.training_2/Adam/batch_normalization_10/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training_2/Adam/batch_normalization_10/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_2/Adam/conv2d_11/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/conv2d_11/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.training_2/Adam/batch_normalization_11/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training_2/Adam/batch_normalization_11/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_2/Adam/conv2d_12/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/conv2d_12/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.training_2/Adam/batch_normalization_12/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training_2/Adam/batch_normalization_12/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_2/Adam/conv2d_13/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/conv2d_13/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.training_2/Adam/batch_normalization_13/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training_2/Adam/batch_normalization_13/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_2/Adam/conv2d_14/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/conv2d_14/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.training_2/Adam/batch_normalization_14/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training_2/Adam/batch_normalization_14/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/dense_6/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_2/Adam/dense_6/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/dense_7/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_2/Adam/dense_7/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/dense_8/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_2/Adam/dense_8/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_2/Adam/conv2d_10/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/conv2d_10/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.training_2/Adam/batch_normalization_10/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training_2/Adam/batch_normalization_10/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_2/Adam/conv2d_11/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/conv2d_11/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.training_2/Adam/batch_normalization_11/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training_2/Adam/batch_normalization_11/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_2/Adam/conv2d_12/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/conv2d_12/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.training_2/Adam/batch_normalization_12/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training_2/Adam/batch_normalization_12/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_2/Adam/conv2d_13/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/conv2d_13/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.training_2/Adam/batch_normalization_13/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training_2/Adam/batch_normalization_13/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_2/Adam/conv2d_14/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/conv2d_14/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.training_2/Adam/batch_normalization_14/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training_2/Adam/batch_normalization_14/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/dense_6/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_2/Adam/dense_6/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/dense_7/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_2/Adam/dense_7/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_2/Adam/dense_8/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_2/Adam/dense_8/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_2Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_10/kernelconv2d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variancedense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias*0
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
"__inference_signature_wrapper_6831
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?*
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp(training_2/Adam/iter/Read/ReadVariableOp*training_2/Adam/beta_1/Read/ReadVariableOp*training_2/Adam/beta_2/Read/ReadVariableOp)training_2/Adam/decay/Read/ReadVariableOp1training_2/Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp6training_2/Adam/conv2d_10/kernel/m/Read/ReadVariableOp4training_2/Adam/conv2d_10/bias/m/Read/ReadVariableOpBtraining_2/Adam/batch_normalization_10/gamma/m/Read/ReadVariableOpAtraining_2/Adam/batch_normalization_10/beta/m/Read/ReadVariableOp6training_2/Adam/conv2d_11/kernel/m/Read/ReadVariableOp4training_2/Adam/conv2d_11/bias/m/Read/ReadVariableOpBtraining_2/Adam/batch_normalization_11/gamma/m/Read/ReadVariableOpAtraining_2/Adam/batch_normalization_11/beta/m/Read/ReadVariableOp6training_2/Adam/conv2d_12/kernel/m/Read/ReadVariableOp4training_2/Adam/conv2d_12/bias/m/Read/ReadVariableOpBtraining_2/Adam/batch_normalization_12/gamma/m/Read/ReadVariableOpAtraining_2/Adam/batch_normalization_12/beta/m/Read/ReadVariableOp6training_2/Adam/conv2d_13/kernel/m/Read/ReadVariableOp4training_2/Adam/conv2d_13/bias/m/Read/ReadVariableOpBtraining_2/Adam/batch_normalization_13/gamma/m/Read/ReadVariableOpAtraining_2/Adam/batch_normalization_13/beta/m/Read/ReadVariableOp6training_2/Adam/conv2d_14/kernel/m/Read/ReadVariableOp4training_2/Adam/conv2d_14/bias/m/Read/ReadVariableOpBtraining_2/Adam/batch_normalization_14/gamma/m/Read/ReadVariableOpAtraining_2/Adam/batch_normalization_14/beta/m/Read/ReadVariableOp4training_2/Adam/dense_6/kernel/m/Read/ReadVariableOp2training_2/Adam/dense_6/bias/m/Read/ReadVariableOp4training_2/Adam/dense_7/kernel/m/Read/ReadVariableOp2training_2/Adam/dense_7/bias/m/Read/ReadVariableOp4training_2/Adam/dense_8/kernel/m/Read/ReadVariableOp2training_2/Adam/dense_8/bias/m/Read/ReadVariableOp6training_2/Adam/conv2d_10/kernel/v/Read/ReadVariableOp4training_2/Adam/conv2d_10/bias/v/Read/ReadVariableOpBtraining_2/Adam/batch_normalization_10/gamma/v/Read/ReadVariableOpAtraining_2/Adam/batch_normalization_10/beta/v/Read/ReadVariableOp6training_2/Adam/conv2d_11/kernel/v/Read/ReadVariableOp4training_2/Adam/conv2d_11/bias/v/Read/ReadVariableOpBtraining_2/Adam/batch_normalization_11/gamma/v/Read/ReadVariableOpAtraining_2/Adam/batch_normalization_11/beta/v/Read/ReadVariableOp6training_2/Adam/conv2d_12/kernel/v/Read/ReadVariableOp4training_2/Adam/conv2d_12/bias/v/Read/ReadVariableOpBtraining_2/Adam/batch_normalization_12/gamma/v/Read/ReadVariableOpAtraining_2/Adam/batch_normalization_12/beta/v/Read/ReadVariableOp6training_2/Adam/conv2d_13/kernel/v/Read/ReadVariableOp4training_2/Adam/conv2d_13/bias/v/Read/ReadVariableOpBtraining_2/Adam/batch_normalization_13/gamma/v/Read/ReadVariableOpAtraining_2/Adam/batch_normalization_13/beta/v/Read/ReadVariableOp6training_2/Adam/conv2d_14/kernel/v/Read/ReadVariableOp4training_2/Adam/conv2d_14/bias/v/Read/ReadVariableOpBtraining_2/Adam/batch_normalization_14/gamma/v/Read/ReadVariableOpAtraining_2/Adam/batch_normalization_14/beta/v/Read/ReadVariableOp4training_2/Adam/dense_6/kernel/v/Read/ReadVariableOp2training_2/Adam/dense_6/bias/v/Read/ReadVariableOp4training_2/Adam/dense_7/kernel/v/Read/ReadVariableOp2training_2/Adam/dense_7/bias/v/Read/ReadVariableOp4training_2/Adam/dense_8/kernel/v/Read/ReadVariableOp2training_2/Adam/dense_8/bias/v/Read/ReadVariableOpConst*l
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
__inference__traced_save_8252
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10/kernelconv2d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variancedense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biastraining_2/Adam/itertraining_2/Adam/beta_1training_2/Adam/beta_2training_2/Adam/decaytraining_2/Adam/learning_ratetotal_1count_1"training_2/Adam/conv2d_10/kernel/m training_2/Adam/conv2d_10/bias/m.training_2/Adam/batch_normalization_10/gamma/m-training_2/Adam/batch_normalization_10/beta/m"training_2/Adam/conv2d_11/kernel/m training_2/Adam/conv2d_11/bias/m.training_2/Adam/batch_normalization_11/gamma/m-training_2/Adam/batch_normalization_11/beta/m"training_2/Adam/conv2d_12/kernel/m training_2/Adam/conv2d_12/bias/m.training_2/Adam/batch_normalization_12/gamma/m-training_2/Adam/batch_normalization_12/beta/m"training_2/Adam/conv2d_13/kernel/m training_2/Adam/conv2d_13/bias/m.training_2/Adam/batch_normalization_13/gamma/m-training_2/Adam/batch_normalization_13/beta/m"training_2/Adam/conv2d_14/kernel/m training_2/Adam/conv2d_14/bias/m.training_2/Adam/batch_normalization_14/gamma/m-training_2/Adam/batch_normalization_14/beta/m training_2/Adam/dense_6/kernel/mtraining_2/Adam/dense_6/bias/m training_2/Adam/dense_7/kernel/mtraining_2/Adam/dense_7/bias/m training_2/Adam/dense_8/kernel/mtraining_2/Adam/dense_8/bias/m"training_2/Adam/conv2d_10/kernel/v training_2/Adam/conv2d_10/bias/v.training_2/Adam/batch_normalization_10/gamma/v-training_2/Adam/batch_normalization_10/beta/v"training_2/Adam/conv2d_11/kernel/v training_2/Adam/conv2d_11/bias/v.training_2/Adam/batch_normalization_11/gamma/v-training_2/Adam/batch_normalization_11/beta/v"training_2/Adam/conv2d_12/kernel/v training_2/Adam/conv2d_12/bias/v.training_2/Adam/batch_normalization_12/gamma/v-training_2/Adam/batch_normalization_12/beta/v"training_2/Adam/conv2d_13/kernel/v training_2/Adam/conv2d_13/bias/v.training_2/Adam/batch_normalization_13/gamma/v-training_2/Adam/batch_normalization_13/beta/v"training_2/Adam/conv2d_14/kernel/v training_2/Adam/conv2d_14/bias/v.training_2/Adam/batch_normalization_14/gamma/v-training_2/Adam/batch_normalization_14/beta/v training_2/Adam/dense_6/kernel/vtraining_2/Adam/dense_6/bias/v training_2/Adam/dense_7/kernel/vtraining_2/Adam/dense_7/bias/v training_2/Adam/dense_8/kernel/vtraining_2/Adam/dense_8/bias/v*k
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
 __inference__traced_restore_8547??
?

?
5__inference_batch_normalization_12_layer_call_fn_7474

inputs*
batch_normalization_12_gamma:0)
batch_normalization_12_beta:00
"batch_normalization_12_moving_mean:04
&batch_normalization_12_moving_variance:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_50612
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
?
?
&__inference_dense_7_layer_call_fn_7888

inputs!
dense_7_kernel:	?@
dense_7_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_kerneldense_7_bias*
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
A__inference_dense_7_layer_call_and_return_conditional_losses_57152
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
??
?0
__inference__traced_save_8252
file_prefix/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop3
/savev2_training_2_adam_iter_read_readvariableop	5
1savev2_training_2_adam_beta_1_read_readvariableop5
1savev2_training_2_adam_beta_2_read_readvariableop4
0savev2_training_2_adam_decay_read_readvariableop<
8savev2_training_2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopA
=savev2_training_2_adam_conv2d_10_kernel_m_read_readvariableop?
;savev2_training_2_adam_conv2d_10_bias_m_read_readvariableopM
Isavev2_training_2_adam_batch_normalization_10_gamma_m_read_readvariableopL
Hsavev2_training_2_adam_batch_normalization_10_beta_m_read_readvariableopA
=savev2_training_2_adam_conv2d_11_kernel_m_read_readvariableop?
;savev2_training_2_adam_conv2d_11_bias_m_read_readvariableopM
Isavev2_training_2_adam_batch_normalization_11_gamma_m_read_readvariableopL
Hsavev2_training_2_adam_batch_normalization_11_beta_m_read_readvariableopA
=savev2_training_2_adam_conv2d_12_kernel_m_read_readvariableop?
;savev2_training_2_adam_conv2d_12_bias_m_read_readvariableopM
Isavev2_training_2_adam_batch_normalization_12_gamma_m_read_readvariableopL
Hsavev2_training_2_adam_batch_normalization_12_beta_m_read_readvariableopA
=savev2_training_2_adam_conv2d_13_kernel_m_read_readvariableop?
;savev2_training_2_adam_conv2d_13_bias_m_read_readvariableopM
Isavev2_training_2_adam_batch_normalization_13_gamma_m_read_readvariableopL
Hsavev2_training_2_adam_batch_normalization_13_beta_m_read_readvariableopA
=savev2_training_2_adam_conv2d_14_kernel_m_read_readvariableop?
;savev2_training_2_adam_conv2d_14_bias_m_read_readvariableopM
Isavev2_training_2_adam_batch_normalization_14_gamma_m_read_readvariableopL
Hsavev2_training_2_adam_batch_normalization_14_beta_m_read_readvariableop?
;savev2_training_2_adam_dense_6_kernel_m_read_readvariableop=
9savev2_training_2_adam_dense_6_bias_m_read_readvariableop?
;savev2_training_2_adam_dense_7_kernel_m_read_readvariableop=
9savev2_training_2_adam_dense_7_bias_m_read_readvariableop?
;savev2_training_2_adam_dense_8_kernel_m_read_readvariableop=
9savev2_training_2_adam_dense_8_bias_m_read_readvariableopA
=savev2_training_2_adam_conv2d_10_kernel_v_read_readvariableop?
;savev2_training_2_adam_conv2d_10_bias_v_read_readvariableopM
Isavev2_training_2_adam_batch_normalization_10_gamma_v_read_readvariableopL
Hsavev2_training_2_adam_batch_normalization_10_beta_v_read_readvariableopA
=savev2_training_2_adam_conv2d_11_kernel_v_read_readvariableop?
;savev2_training_2_adam_conv2d_11_bias_v_read_readvariableopM
Isavev2_training_2_adam_batch_normalization_11_gamma_v_read_readvariableopL
Hsavev2_training_2_adam_batch_normalization_11_beta_v_read_readvariableopA
=savev2_training_2_adam_conv2d_12_kernel_v_read_readvariableop?
;savev2_training_2_adam_conv2d_12_bias_v_read_readvariableopM
Isavev2_training_2_adam_batch_normalization_12_gamma_v_read_readvariableopL
Hsavev2_training_2_adam_batch_normalization_12_beta_v_read_readvariableopA
=savev2_training_2_adam_conv2d_13_kernel_v_read_readvariableop?
;savev2_training_2_adam_conv2d_13_bias_v_read_readvariableopM
Isavev2_training_2_adam_batch_normalization_13_gamma_v_read_readvariableopL
Hsavev2_training_2_adam_batch_normalization_13_beta_v_read_readvariableopA
=savev2_training_2_adam_conv2d_14_kernel_v_read_readvariableop?
;savev2_training_2_adam_conv2d_14_bias_v_read_readvariableopM
Isavev2_training_2_adam_batch_normalization_14_gamma_v_read_readvariableopL
Hsavev2_training_2_adam_batch_normalization_14_beta_v_read_readvariableop?
;savev2_training_2_adam_dense_6_kernel_v_read_readvariableop=
9savev2_training_2_adam_dense_6_bias_v_read_readvariableop?
;savev2_training_2_adam_dense_7_kernel_v_read_readvariableop=
9savev2_training_2_adam_dense_7_bias_v_read_readvariableop?
;savev2_training_2_adam_dense_8_kernel_v_read_readvariableop=
9savev2_training_2_adam_dense_8_bias_v_read_readvariableop
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
SaveV2/shape_and_slices?.
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop/savev2_training_2_adam_iter_read_readvariableop1savev2_training_2_adam_beta_1_read_readvariableop1savev2_training_2_adam_beta_2_read_readvariableop0savev2_training_2_adam_decay_read_readvariableop8savev2_training_2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop=savev2_training_2_adam_conv2d_10_kernel_m_read_readvariableop;savev2_training_2_adam_conv2d_10_bias_m_read_readvariableopIsavev2_training_2_adam_batch_normalization_10_gamma_m_read_readvariableopHsavev2_training_2_adam_batch_normalization_10_beta_m_read_readvariableop=savev2_training_2_adam_conv2d_11_kernel_m_read_readvariableop;savev2_training_2_adam_conv2d_11_bias_m_read_readvariableopIsavev2_training_2_adam_batch_normalization_11_gamma_m_read_readvariableopHsavev2_training_2_adam_batch_normalization_11_beta_m_read_readvariableop=savev2_training_2_adam_conv2d_12_kernel_m_read_readvariableop;savev2_training_2_adam_conv2d_12_bias_m_read_readvariableopIsavev2_training_2_adam_batch_normalization_12_gamma_m_read_readvariableopHsavev2_training_2_adam_batch_normalization_12_beta_m_read_readvariableop=savev2_training_2_adam_conv2d_13_kernel_m_read_readvariableop;savev2_training_2_adam_conv2d_13_bias_m_read_readvariableopIsavev2_training_2_adam_batch_normalization_13_gamma_m_read_readvariableopHsavev2_training_2_adam_batch_normalization_13_beta_m_read_readvariableop=savev2_training_2_adam_conv2d_14_kernel_m_read_readvariableop;savev2_training_2_adam_conv2d_14_bias_m_read_readvariableopIsavev2_training_2_adam_batch_normalization_14_gamma_m_read_readvariableopHsavev2_training_2_adam_batch_normalization_14_beta_m_read_readvariableop;savev2_training_2_adam_dense_6_kernel_m_read_readvariableop9savev2_training_2_adam_dense_6_bias_m_read_readvariableop;savev2_training_2_adam_dense_7_kernel_m_read_readvariableop9savev2_training_2_adam_dense_7_bias_m_read_readvariableop;savev2_training_2_adam_dense_8_kernel_m_read_readvariableop9savev2_training_2_adam_dense_8_bias_m_read_readvariableop=savev2_training_2_adam_conv2d_10_kernel_v_read_readvariableop;savev2_training_2_adam_conv2d_10_bias_v_read_readvariableopIsavev2_training_2_adam_batch_normalization_10_gamma_v_read_readvariableopHsavev2_training_2_adam_batch_normalization_10_beta_v_read_readvariableop=savev2_training_2_adam_conv2d_11_kernel_v_read_readvariableop;savev2_training_2_adam_conv2d_11_bias_v_read_readvariableopIsavev2_training_2_adam_batch_normalization_11_gamma_v_read_readvariableopHsavev2_training_2_adam_batch_normalization_11_beta_v_read_readvariableop=savev2_training_2_adam_conv2d_12_kernel_v_read_readvariableop;savev2_training_2_adam_conv2d_12_bias_v_read_readvariableopIsavev2_training_2_adam_batch_normalization_12_gamma_v_read_readvariableopHsavev2_training_2_adam_batch_normalization_12_beta_v_read_readvariableop=savev2_training_2_adam_conv2d_13_kernel_v_read_readvariableop;savev2_training_2_adam_conv2d_13_bias_v_read_readvariableopIsavev2_training_2_adam_batch_normalization_13_gamma_v_read_readvariableopHsavev2_training_2_adam_batch_normalization_13_beta_v_read_readvariableop=savev2_training_2_adam_conv2d_14_kernel_v_read_readvariableop;savev2_training_2_adam_conv2d_14_bias_v_read_readvariableopIsavev2_training_2_adam_batch_normalization_14_gamma_v_read_readvariableopHsavev2_training_2_adam_batch_normalization_14_beta_v_read_readvariableop;savev2_training_2_adam_dense_6_kernel_v_read_readvariableop9savev2_training_2_adam_dense_6_bias_v_read_readvariableop;savev2_training_2_adam_dense_7_kernel_v_read_readvariableop9savev2_training_2_adam_dense_7_bias_v_read_readvariableop;savev2_training_2_adam_dense_8_kernel_v_read_readvariableop9savev2_training_2_adam_dense_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: ::::::: : : : : : : 0:0:0:0:0:0:0@:@:@:@:@:@:@@:@:@:@:@:@:???:?:	?@:@:@:: : : : : : : ::::: : : : : 0:0:0:0:0@:@:@:@:@@:@:@:@:???:?:	?@:@:@:::::: : : : : 0:0:0:0:0@:@:@:@:@@:@:@:@:???:?:	?@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 
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
:: -
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
:: G
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
?

?
5__inference_batch_normalization_12_layer_call_fn_7501

inputs*
batch_normalization_12_gamma:0)
batch_normalization_12_beta:00
"batch_normalization_12_moving_mean:04
&batch_normalization_12_moving_variance:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_61412
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
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_5668

inputs9
+readvariableop_batch_normalization_14_gamma:@:
,readvariableop_1_batch_normalization_14_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_14_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_14_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance*
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
?

?
5__inference_batch_normalization_12_layer_call_fn_7492

inputs*
batch_normalization_12_gamma:0)
batch_normalization_12_beta:00
"batch_normalization_12_moving_mean:04
&batch_normalization_12_moving_variance:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_55922
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
?
?
A__inference_dense_7_layer_call_and_return_conditional_losses_5715

inputs7
$matmul_readvariableop_dense_7_kernel:	?@1
#biasadd_readvariableop_dense_7_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_7_kernel*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_7_bias*
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
C__inference_conv2d_12_layer_call_and_return_conditional_losses_5571

inputs@
&conv2d_readvariableop_conv2d_12_kernel: 03
%biasadd_readvariableop_conv2d_12_bias:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_12_kernel*&
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
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_12_bias*
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
?
?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7519

inputs9
+readvariableop_batch_normalization_12_gamma:0:
,readvariableop_1_batch_normalization_12_beta:0P
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean:0V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
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
?

?
5__inference_batch_normalization_13_layer_call_fn_7627

inputs*
batch_normalization_13_gamma:@)
batch_normalization_13_beta:@0
"batch_normalization_13_moving_mean:@4
&batch_normalization_13_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_13_gammabatch_normalization_13_beta"batch_normalization_13_moving_mean&batch_normalization_13_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_60502
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
?
?
C__inference_conv2d_13_layer_call_and_return_conditional_losses_5609

inputs@
&conv2d_readvariableop_conv2d_13_kernel:0@3
%biasadd_readvariableop_conv2d_13_bias:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_13_kernel*&
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
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_13_bias*
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
?
?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7537

inputs9
+readvariableop_batch_normalization_12_gamma:0:
,readvariableop_1_batch_normalization_12_beta:0P
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean:0V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_4769

inputs9
+readvariableop_batch_normalization_10_gamma::
,readvariableop_1_batch_normalization_10_beta:P
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean:V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
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
?
?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7573

inputs9
+readvariableop_batch_normalization_12_gamma:0:
,readvariableop_1_batch_normalization_12_beta:0P
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean:0V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?

?
5__inference_batch_normalization_11_layer_call_fn_7375

inputs*
batch_normalization_11_gamma: )
batch_normalization_11_beta: 0
"batch_normalization_11_moving_mean: 4
&batch_normalization_11_moving_variance: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_62322
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
?'
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_7195

inputsJ
0conv2d_10_conv2d_readvariableop_conv2d_10_kernel:=
/conv2d_10_biasadd_readvariableop_conv2d_10_bias:P
Bbatch_normalization_10_readvariableop_batch_normalization_10_gamma:Q
Cbatch_normalization_10_readvariableop_1_batch_normalization_10_beta:g
Ybatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean:m
_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance:J
0conv2d_11_conv2d_readvariableop_conv2d_11_kernel: =
/conv2d_11_biasadd_readvariableop_conv2d_11_bias: P
Bbatch_normalization_11_readvariableop_batch_normalization_11_gamma: Q
Cbatch_normalization_11_readvariableop_1_batch_normalization_11_beta: g
Ybatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean: m
_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance: J
0conv2d_12_conv2d_readvariableop_conv2d_12_kernel: 0=
/conv2d_12_biasadd_readvariableop_conv2d_12_bias:0P
Bbatch_normalization_12_readvariableop_batch_normalization_12_gamma:0Q
Cbatch_normalization_12_readvariableop_1_batch_normalization_12_beta:0g
Ybatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean:0m
_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance:0J
0conv2d_13_conv2d_readvariableop_conv2d_13_kernel:0@=
/conv2d_13_biasadd_readvariableop_conv2d_13_bias:@P
Bbatch_normalization_13_readvariableop_batch_normalization_13_gamma:@Q
Cbatch_normalization_13_readvariableop_1_batch_normalization_13_beta:@g
Ybatch_normalization_13_fusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean:@m
_batch_normalization_13_fusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance:@J
0conv2d_14_conv2d_readvariableop_conv2d_14_kernel:@@=
/conv2d_14_biasadd_readvariableop_conv2d_14_bias:@P
Bbatch_normalization_14_readvariableop_batch_normalization_14_gamma:@Q
Cbatch_normalization_14_readvariableop_1_batch_normalization_14_beta:@g
Ybatch_normalization_14_fusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean:@m
_batch_normalization_14_fusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance:@A
,dense_6_matmul_readvariableop_dense_6_kernel:???:
+dense_6_biasadd_readvariableop_dense_6_bias:	??
,dense_7_matmul_readvariableop_dense_7_kernel:	?@9
+dense_7_biasadd_readvariableop_dense_7_bias:@>
,dense_8_matmul_readvariableop_dense_8_kernel:@9
+dense_8_biasadd_readvariableop_dense_8_bias:
identity??%batch_normalization_10/AssignNewValue?'batch_normalization_10/AssignNewValue_1?6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?%batch_normalization_11/AssignNewValue?'batch_normalization_11/AssignNewValue_1?6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_11/ReadVariableOp?'batch_normalization_11/ReadVariableOp_1?%batch_normalization_12/AssignNewValue?'batch_normalization_12/AssignNewValue_1?6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?%batch_normalization_13/AssignNewValue?'batch_normalization_13/AssignNewValue_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?%batch_normalization_14/AssignNewValue?'batch_normalization_14/AssignNewValue_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp0conv2d_10_conv2d_readvariableop_conv2d_10_kernel*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp/conv2d_10_biasadd_readvariableop_conv2d_10_bias*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_10/BiasAdd?
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_10/Relu?
%batch_normalization_10/ReadVariableOpReadVariableOpBbatch_normalization_10_readvariableop_batch_normalization_10_gamma*
_output_shapes
:*
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOpCbatch_normalization_10_readvariableop_1_batch_normalization_10_beta*
_output_shapes
:*
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_10/FusedBatchNormV3?
%batch_normalization_10/AssignNewValueAssignVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValue?
'batch_normalization_10/AssignNewValue_1AssignVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp0conv2d_11_conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
: *
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH *
paddingVALID*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp/conv2d_11_biasadd_readvariableop_conv2d_11_bias*
_output_shapes
: *
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH 2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????HH 2
conv2d_11/Relu?
%batch_normalization_11/ReadVariableOpReadVariableOpBbatch_normalization_11_readvariableop_batch_normalization_11_gamma*
_output_shapes
: *
dtype02'
%batch_normalization_11/ReadVariableOp?
'batch_normalization_11/ReadVariableOp_1ReadVariableOpCbatch_normalization_11_readvariableop_1_batch_normalization_11_beta*
_output_shapes
: *
dtype02)
'batch_normalization_11/ReadVariableOp_1?
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
: *
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
: *
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????HH : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_11/FusedBatchNormV3?
%batch_normalization_11/AssignNewValueAssignVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_11/AssignNewValue?
'batch_normalization_11/AssignNewValue_1AssignVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_11/AssignNewValue_1?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp0conv2d_12_conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: 0*
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D+batch_normalization_11/FusedBatchNormV3:y:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##0*
paddingVALID*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp/conv2d_12_biasadd_readvariableop_conv2d_12_bias*
_output_shapes
:0*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##02
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????##02
conv2d_12/Relu?
%batch_normalization_12/ReadVariableOpReadVariableOpBbatch_normalization_12_readvariableop_batch_normalization_12_gamma*
_output_shapes
:0*
dtype02'
%batch_normalization_12/ReadVariableOp?
'batch_normalization_12/ReadVariableOp_1ReadVariableOpCbatch_normalization_12_readvariableop_1_batch_normalization_12_beta*
_output_shapes
:0*
dtype02)
'batch_normalization_12/ReadVariableOp_1?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:0*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:0*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_12/Relu:activations:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????##0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_12/FusedBatchNormV3?
%batch_normalization_12/AssignNewValueAssignVariableOpYbatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_12/AssignNewValue?
'batch_normalization_12/AssignNewValue_1AssignVariableOp_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_12/AssignNewValue_1?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp0conv2d_13_conv2d_readvariableop_conv2d_13_kernel*&
_output_shapes
:0@*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D+batch_normalization_12/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@*
paddingVALID*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp/conv2d_13_biasadd_readvariableop_conv2d_13_bias*
_output_shapes
:@*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!!@2
conv2d_13/Relu?
%batch_normalization_13/ReadVariableOpReadVariableOpBbatch_normalization_13_readvariableop_batch_normalization_13_gamma*
_output_shapes
:@*
dtype02'
%batch_normalization_13/ReadVariableOp?
'batch_normalization_13/ReadVariableOp_1ReadVariableOpCbatch_normalization_13_readvariableop_1_batch_normalization_13_beta*
_output_shapes
:@*
dtype02)
'batch_normalization_13/ReadVariableOp_1?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_13_fusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean*
_output_shapes
:@*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_13_fusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance*
_output_shapes
:@*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????!!@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_13/FusedBatchNormV3?
%batch_normalization_13/AssignNewValueAssignVariableOpYbatch_normalization_13_fusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_13/AssignNewValue?
'batch_normalization_13/AssignNewValue_1AssignVariableOp_batch_normalization_13_fusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_13/AssignNewValue_1?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp0conv2d_14_conv2d_readvariableop_conv2d_14_kernel*&
_output_shapes
:@@*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D+batch_normalization_13/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp/conv2d_14_biasadd_readvariableop_conv2d_14_bias*
_output_shapes
:@*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/BiasAdd~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/Relu?
%batch_normalization_14/ReadVariableOpReadVariableOpBbatch_normalization_14_readvariableop_batch_normalization_14_gamma*
_output_shapes
:@*
dtype02'
%batch_normalization_14/ReadVariableOp?
'batch_normalization_14/ReadVariableOp_1ReadVariableOpCbatch_normalization_14_readvariableop_1_batch_normalization_14_beta*
_output_shapes
:@*
dtype02)
'batch_normalization_14/ReadVariableOp_1?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_14_fusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean*
_output_shapes
:@*
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_14_fusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance*
_output_shapes
:@*
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_14/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_14/FusedBatchNormV3?
%batch_normalization_14/AssignNewValueAssignVariableOpYbatch_normalization_14_fusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean4batch_normalization_14/FusedBatchNormV3:batch_mean:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_14/AssignNewValue?
'batch_normalization_14/AssignNewValue_1AssignVariableOp_batch_normalization_14_fusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance8batch_normalization_14/FusedBatchNormV3:batch_variance:09^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_14/AssignNewValue_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@?  2
flatten_2/Const?
flatten_2/ReshapeReshape+batch_normalization_14/FusedBatchNormV3:y:0flatten_2/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_2/Reshape?
dense_6/MatMul/ReadVariableOpReadVariableOp,dense_6_matmul_readvariableop_dense_6_kernel*!
_output_shapes
:???*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp+dense_6_biasadd_readvariableop_dense_6_bias*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Reluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_4/dropout/Const?
dropout_4/dropout/MulMuldense_6/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/dropout/Mul|
dropout_4/dropout/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_4/dropout/Mul_1?
dense_7/MatMul/ReadVariableOpReadVariableOp,dense_7_matmul_readvariableop_dense_7_kernel*
_output_shapes
:	?@*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp+dense_7_biasadd_readvariableop_dense_7_bias*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_7/Reluw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_5/dropout/Const?
dropout_5/dropout/MulMuldense_7/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout_5/dropout/Mul|
dropout_5/dropout/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout_5/dropout/Mul_1?
dense_8/MatMul/ReadVariableOpReadVariableOp,dense_8_matmul_readvariableop_dense_8_kernel*
_output_shapes

:@*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp+dense_8_biasadd_readvariableop_dense_8_bias*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAddy
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_8/Sigmoidn
IdentityIdentitydense_8/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_1&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1&^batch_normalization_14/AssignNewValue(^batch_normalization_14/AssignNewValue_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12N
%batch_normalization_14/AssignNewValue%batch_normalization_14/AssignNewValue2R
'batch_normalization_14/AssignNewValue_1'batch_normalization_14/AssignNewValue_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_5_layer_call_fn_7904

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
C__inference_dropout_5_layer_call_and_return_conditional_losses_57242
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
?
a
C__inference_dropout_5_layer_call_and_return_conditional_losses_5724

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
?
D
(__inference_flatten_2_layer_call_fn_7830

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
GPU2*0J 8? *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_56802
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
?
D
(__inference_dropout_4_layer_call_fn_7859

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
GPU2*0J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_57022
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
?

?
5__inference_batch_normalization_10_layer_call_fn_7249

inputs*
batch_normalization_10_gamma:)
batch_normalization_10_beta:0
"batch_normalization_10_moving_mean:4
&batch_normalization_10_moving_variance:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_63232
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
?
?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7555

inputs9
+readvariableop_batch_normalization_12_gamma:0:
,readvariableop_1_batch_normalization_12_beta:0P
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean:0V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
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
?
?
(__inference_conv2d_10_layer_call_fn_7202

inputs*
conv2d_10_kernel:
conv2d_10_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_kernelconv2d_10_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_54952
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
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_11_layer_call_and_return_conditional_losses_7339

inputs@
&conv2d_readvariableop_conv2d_11_kernel: 3
%biasadd_readvariableop_conv2d_11_bias: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_11_kernel*&
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
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_11_bias*
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
(__inference_conv2d_13_layer_call_fn_7580

inputs*
conv2d_13_kernel:0@
conv2d_13_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_13_kernelconv2d_13_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_56092
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
?

?
5__inference_batch_normalization_14_layer_call_fn_7726

inputs*
batch_normalization_14_gamma:@)
batch_normalization_14_beta:@0
"batch_normalization_14_moving_mean:@4
&batch_normalization_14_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_14_gammabatch_normalization_14_beta"batch_normalization_14_moving_mean&batch_normalization_14_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_53532
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
?
a
(__inference_dropout_5_layer_call_fn_7909

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
C__inference_dropout_5_layer_call_and_return_conditional_losses_58202
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
(__inference_conv2d_12_layer_call_fn_7454

inputs*
conv2d_12_kernel: 0
conv2d_12_bias:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_kernelconv2d_12_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_55712
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
?
?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_5592

inputs9
+readvariableop_batch_normalization_12_gamma:0:
,readvariableop_1_batch_normalization_12_beta:0P
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean:0V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
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
?
b
C__inference_dropout_4_layer_call_and_return_conditional_losses_5879

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
?
?
(__inference_conv2d_11_layer_call_fn_7328

inputs*
conv2d_11_kernel: 
conv2d_11_bias: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_kernelconv2d_11_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_55332
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
?

?
A__inference_dense_6_layer_call_and_return_conditional_losses_7854

inputs9
$matmul_readvariableop_dense_6_kernel:???2
#biasadd_readvariableop_dense_6_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_6_kernel*!
_output_shapes
:???*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_6_bias*
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
?
?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7429

inputs9
+readvariableop_batch_normalization_11_gamma: :
,readvariableop_1_batch_normalization_11_beta: P
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean: V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
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
?
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_7836

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
?
?
8__inference_single_image_simple_model_layer_call_fn_6872

inputs*
conv2d_10_kernel:
conv2d_10_bias:*
batch_normalization_10_gamma:)
batch_normalization_10_beta:0
"batch_normalization_10_moving_mean:4
&batch_normalization_10_moving_variance:*
conv2d_11_kernel: 
conv2d_11_bias: *
batch_normalization_11_gamma: )
batch_normalization_11_beta: 0
"batch_normalization_11_moving_mean: 4
&batch_normalization_11_moving_variance: *
conv2d_12_kernel: 0
conv2d_12_bias:0*
batch_normalization_12_gamma:0)
batch_normalization_12_beta:00
"batch_normalization_12_moving_mean:04
&batch_normalization_12_moving_variance:0*
conv2d_13_kernel:0@
conv2d_13_bias:@*
batch_normalization_13_gamma:@)
batch_normalization_13_beta:@0
"batch_normalization_13_moving_mean:@4
&batch_normalization_13_moving_variance:@*
conv2d_14_kernel:@@
conv2d_14_bias:@*
batch_normalization_14_gamma:@)
batch_normalization_14_beta:@0
"batch_normalization_14_moving_mean:@4
&batch_normalization_14_moving_variance:@#
dense_6_kernel:???
dense_6_bias:	?!
dense_7_kernel:	?@
dense_7_bias:@ 
dense_8_kernel:@
dense_8_bias:
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_kernelconv2d_10_biasbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_varianceconv2d_11_kernelconv2d_11_biasbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_varianceconv2d_12_kernelconv2d_12_biasbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_varianceconv2d_13_kernelconv2d_13_biasbatch_normalization_13_gammabatch_normalization_13_beta"batch_normalization_13_moving_mean&batch_normalization_13_moving_varianceconv2d_14_kernelconv2d_14_biasbatch_normalization_14_gammabatch_normalization_14_beta"batch_normalization_14_moving_mean&batch_normalization_14_moving_variancedense_6_kerneldense_6_biasdense_7_kerneldense_7_biasdense_8_kerneldense_8_bias*0
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
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_57422
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
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?C
 __inference__traced_restore_8547
file_prefix;
!assignvariableop_conv2d_10_kernel:/
!assignvariableop_1_conv2d_10_bias:=
/assignvariableop_2_batch_normalization_10_gamma:<
.assignvariableop_3_batch_normalization_10_beta:C
5assignvariableop_4_batch_normalization_10_moving_mean:G
9assignvariableop_5_batch_normalization_10_moving_variance:=
#assignvariableop_6_conv2d_11_kernel: /
!assignvariableop_7_conv2d_11_bias: =
/assignvariableop_8_batch_normalization_11_gamma: <
.assignvariableop_9_batch_normalization_11_beta: D
6assignvariableop_10_batch_normalization_11_moving_mean: H
:assignvariableop_11_batch_normalization_11_moving_variance: >
$assignvariableop_12_conv2d_12_kernel: 00
"assignvariableop_13_conv2d_12_bias:0>
0assignvariableop_14_batch_normalization_12_gamma:0=
/assignvariableop_15_batch_normalization_12_beta:0D
6assignvariableop_16_batch_normalization_12_moving_mean:0H
:assignvariableop_17_batch_normalization_12_moving_variance:0>
$assignvariableop_18_conv2d_13_kernel:0@0
"assignvariableop_19_conv2d_13_bias:@>
0assignvariableop_20_batch_normalization_13_gamma:@=
/assignvariableop_21_batch_normalization_13_beta:@D
6assignvariableop_22_batch_normalization_13_moving_mean:@H
:assignvariableop_23_batch_normalization_13_moving_variance:@>
$assignvariableop_24_conv2d_14_kernel:@@0
"assignvariableop_25_conv2d_14_bias:@>
0assignvariableop_26_batch_normalization_14_gamma:@=
/assignvariableop_27_batch_normalization_14_beta:@D
6assignvariableop_28_batch_normalization_14_moving_mean:@H
:assignvariableop_29_batch_normalization_14_moving_variance:@7
"assignvariableop_30_dense_6_kernel:???/
 assignvariableop_31_dense_6_bias:	?5
"assignvariableop_32_dense_7_kernel:	?@.
 assignvariableop_33_dense_7_bias:@4
"assignvariableop_34_dense_8_kernel:@.
 assignvariableop_35_dense_8_bias:2
(assignvariableop_36_training_2_adam_iter:	 4
*assignvariableop_37_training_2_adam_beta_1: 4
*assignvariableop_38_training_2_adam_beta_2: 3
)assignvariableop_39_training_2_adam_decay: ;
1assignvariableop_40_training_2_adam_learning_rate: %
assignvariableop_41_total_1: %
assignvariableop_42_count_1: P
6assignvariableop_43_training_2_adam_conv2d_10_kernel_m:B
4assignvariableop_44_training_2_adam_conv2d_10_bias_m:P
Bassignvariableop_45_training_2_adam_batch_normalization_10_gamma_m:O
Aassignvariableop_46_training_2_adam_batch_normalization_10_beta_m:P
6assignvariableop_47_training_2_adam_conv2d_11_kernel_m: B
4assignvariableop_48_training_2_adam_conv2d_11_bias_m: P
Bassignvariableop_49_training_2_adam_batch_normalization_11_gamma_m: O
Aassignvariableop_50_training_2_adam_batch_normalization_11_beta_m: P
6assignvariableop_51_training_2_adam_conv2d_12_kernel_m: 0B
4assignvariableop_52_training_2_adam_conv2d_12_bias_m:0P
Bassignvariableop_53_training_2_adam_batch_normalization_12_gamma_m:0O
Aassignvariableop_54_training_2_adam_batch_normalization_12_beta_m:0P
6assignvariableop_55_training_2_adam_conv2d_13_kernel_m:0@B
4assignvariableop_56_training_2_adam_conv2d_13_bias_m:@P
Bassignvariableop_57_training_2_adam_batch_normalization_13_gamma_m:@O
Aassignvariableop_58_training_2_adam_batch_normalization_13_beta_m:@P
6assignvariableop_59_training_2_adam_conv2d_14_kernel_m:@@B
4assignvariableop_60_training_2_adam_conv2d_14_bias_m:@P
Bassignvariableop_61_training_2_adam_batch_normalization_14_gamma_m:@O
Aassignvariableop_62_training_2_adam_batch_normalization_14_beta_m:@I
4assignvariableop_63_training_2_adam_dense_6_kernel_m:???A
2assignvariableop_64_training_2_adam_dense_6_bias_m:	?G
4assignvariableop_65_training_2_adam_dense_7_kernel_m:	?@@
2assignvariableop_66_training_2_adam_dense_7_bias_m:@F
4assignvariableop_67_training_2_adam_dense_8_kernel_m:@@
2assignvariableop_68_training_2_adam_dense_8_bias_m:P
6assignvariableop_69_training_2_adam_conv2d_10_kernel_v:B
4assignvariableop_70_training_2_adam_conv2d_10_bias_v:P
Bassignvariableop_71_training_2_adam_batch_normalization_10_gamma_v:O
Aassignvariableop_72_training_2_adam_batch_normalization_10_beta_v:P
6assignvariableop_73_training_2_adam_conv2d_11_kernel_v: B
4assignvariableop_74_training_2_adam_conv2d_11_bias_v: P
Bassignvariableop_75_training_2_adam_batch_normalization_11_gamma_v: O
Aassignvariableop_76_training_2_adam_batch_normalization_11_beta_v: P
6assignvariableop_77_training_2_adam_conv2d_12_kernel_v: 0B
4assignvariableop_78_training_2_adam_conv2d_12_bias_v:0P
Bassignvariableop_79_training_2_adam_batch_normalization_12_gamma_v:0O
Aassignvariableop_80_training_2_adam_batch_normalization_12_beta_v:0P
6assignvariableop_81_training_2_adam_conv2d_13_kernel_v:0@B
4assignvariableop_82_training_2_adam_conv2d_13_bias_v:@P
Bassignvariableop_83_training_2_adam_batch_normalization_13_gamma_v:@O
Aassignvariableop_84_training_2_adam_batch_normalization_13_beta_v:@P
6assignvariableop_85_training_2_adam_conv2d_14_kernel_v:@@B
4assignvariableop_86_training_2_adam_conv2d_14_bias_v:@P
Bassignvariableop_87_training_2_adam_batch_normalization_14_gamma_v:@O
Aassignvariableop_88_training_2_adam_batch_normalization_14_beta_v:@I
4assignvariableop_89_training_2_adam_dense_6_kernel_v:???A
2assignvariableop_90_training_2_adam_dense_6_bias_v:	?G
4assignvariableop_91_training_2_adam_dense_7_kernel_v:	?@@
2assignvariableop_92_training_2_adam_dense_7_bias_v:@F
4assignvariableop_93_training_2_adam_dense_8_kernel_v:@@
2assignvariableop_94_training_2_adam_dense_8_bias_v:
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_10_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_10_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_10_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_10_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_11_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_11_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_11_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_11_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_12_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_12_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_12_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_12_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_12_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_12_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_13_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_13_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_13_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_13_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_13_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_13_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_14_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_14_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_14_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_14_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_14_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_14_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp"assignvariableop_30_dense_6_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp assignvariableop_31_dense_6_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense_7_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense_7_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_8_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp assignvariableop_35_dense_8_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_training_2_adam_iterIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_training_2_adam_beta_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp*assignvariableop_38_training_2_adam_beta_2Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp)assignvariableop_39_training_2_adam_decayIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp1assignvariableop_40_training_2_adam_learning_rateIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp6assignvariableop_43_training_2_adam_conv2d_10_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp4assignvariableop_44_training_2_adam_conv2d_10_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpBassignvariableop_45_training_2_adam_batch_normalization_10_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpAassignvariableop_46_training_2_adam_batch_normalization_10_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp6assignvariableop_47_training_2_adam_conv2d_11_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp4assignvariableop_48_training_2_adam_conv2d_11_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpBassignvariableop_49_training_2_adam_batch_normalization_11_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpAassignvariableop_50_training_2_adam_batch_normalization_11_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp6assignvariableop_51_training_2_adam_conv2d_12_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp4assignvariableop_52_training_2_adam_conv2d_12_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpBassignvariableop_53_training_2_adam_batch_normalization_12_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpAassignvariableop_54_training_2_adam_batch_normalization_12_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp6assignvariableop_55_training_2_adam_conv2d_13_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp4assignvariableop_56_training_2_adam_conv2d_13_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpBassignvariableop_57_training_2_adam_batch_normalization_13_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpAassignvariableop_58_training_2_adam_batch_normalization_13_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp6assignvariableop_59_training_2_adam_conv2d_14_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp4assignvariableop_60_training_2_adam_conv2d_14_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpBassignvariableop_61_training_2_adam_batch_normalization_14_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpAassignvariableop_62_training_2_adam_batch_normalization_14_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp4assignvariableop_63_training_2_adam_dense_6_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp2assignvariableop_64_training_2_adam_dense_6_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp4assignvariableop_65_training_2_adam_dense_7_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp2assignvariableop_66_training_2_adam_dense_7_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp4assignvariableop_67_training_2_adam_dense_8_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp2assignvariableop_68_training_2_adam_dense_8_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp6assignvariableop_69_training_2_adam_conv2d_10_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp4assignvariableop_70_training_2_adam_conv2d_10_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOpBassignvariableop_71_training_2_adam_batch_normalization_10_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOpAassignvariableop_72_training_2_adam_batch_normalization_10_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp6assignvariableop_73_training_2_adam_conv2d_11_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp4assignvariableop_74_training_2_adam_conv2d_11_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOpBassignvariableop_75_training_2_adam_batch_normalization_11_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOpAassignvariableop_76_training_2_adam_batch_normalization_11_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp6assignvariableop_77_training_2_adam_conv2d_12_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp4assignvariableop_78_training_2_adam_conv2d_12_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOpBassignvariableop_79_training_2_adam_batch_normalization_12_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOpAassignvariableop_80_training_2_adam_batch_normalization_12_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp6assignvariableop_81_training_2_adam_conv2d_13_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp4assignvariableop_82_training_2_adam_conv2d_13_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOpBassignvariableop_83_training_2_adam_batch_normalization_13_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOpAassignvariableop_84_training_2_adam_batch_normalization_13_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp6assignvariableop_85_training_2_adam_conv2d_14_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp4assignvariableop_86_training_2_adam_conv2d_14_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOpBassignvariableop_87_training_2_adam_batch_normalization_14_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOpAassignvariableop_88_training_2_adam_batch_normalization_14_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp4assignvariableop_89_training_2_adam_dense_6_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp2assignvariableop_90_training_2_adam_dense_6_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp4assignvariableop_91_training_2_adam_dense_7_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp2assignvariableop_92_training_2_adam_dense_7_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp4assignvariableop_93_training_2_adam_dense_8_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp2assignvariableop_94_training_2_adam_dense_8_bias_vIdentity_94:output:0"/device:CPU:0*
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
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6050

inputs9
+readvariableop_batch_normalization_13_gamma:@:
,readvariableop_1_batch_normalization_13_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_13_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_13_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6323

inputs9
+readvariableop_batch_normalization_10_gamma::
,readvariableop_1_batch_normalization_10_beta:P
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean:V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?
?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6232

inputs9
+readvariableop_batch_normalization_11_gamma: :
,readvariableop_1_batch_normalization_11_beta: P
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean: V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
5__inference_batch_normalization_12_layer_call_fn_7483

inputs*
batch_normalization_12_gamma:0)
batch_normalization_12_beta:00
"batch_normalization_12_moving_mean:04
&batch_normalization_12_moving_variance:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_50972
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
?
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5516

inputs9
+readvariableop_batch_normalization_10_gamma::
,readvariableop_1_batch_normalization_10_beta:P
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean:V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
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
5__inference_batch_normalization_10_layer_call_fn_7231

inputs*
batch_normalization_10_gamma:)
batch_normalization_10_beta:0
"batch_normalization_10_moving_mean:4
&batch_normalization_10_moving_variance:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_48052
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
?
?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7411

inputs9
+readvariableop_batch_normalization_11_gamma: :
,readvariableop_1_batch_normalization_11_beta: P
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean: V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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

?
5__inference_batch_normalization_13_layer_call_fn_7600

inputs*
batch_normalization_13_gamma:@)
batch_normalization_13_beta:@0
"batch_normalization_13_moving_mean:@4
&batch_normalization_13_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_13_gammabatch_normalization_13_beta"batch_normalization_13_moving_mean&batch_normalization_13_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_52072
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
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7303

inputs9
+readvariableop_batch_normalization_10_gamma::
,readvariableop_1_batch_normalization_10_beta:P
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean:V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
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
?
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7267

inputs9
+readvariableop_batch_normalization_10_gamma::
,readvariableop_1_batch_normalization_10_beta:P
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean:V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
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

?
A__inference_dense_7_layer_call_and_return_conditional_losses_7899

inputs7
$matmul_readvariableop_dense_7_kernel:	?@1
#biasadd_readvariableop_dense_7_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_7_kernel*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_7_bias*
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
?
?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6141

inputs9
+readvariableop_batch_normalization_12_gamma:0:
,readvariableop_1_batch_normalization_12_beta:0P
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean:0V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?f
?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_6732
input_24
conv2d_10_conv2d_10_kernel:&
conv2d_10_conv2d_10_bias:A
3batch_normalization_10_batch_normalization_10_gamma:@
2batch_normalization_10_batch_normalization_10_beta:G
9batch_normalization_10_batch_normalization_10_moving_mean:K
=batch_normalization_10_batch_normalization_10_moving_variance:4
conv2d_11_conv2d_11_kernel: &
conv2d_11_conv2d_11_bias: A
3batch_normalization_11_batch_normalization_11_gamma: @
2batch_normalization_11_batch_normalization_11_beta: G
9batch_normalization_11_batch_normalization_11_moving_mean: K
=batch_normalization_11_batch_normalization_11_moving_variance: 4
conv2d_12_conv2d_12_kernel: 0&
conv2d_12_conv2d_12_bias:0A
3batch_normalization_12_batch_normalization_12_gamma:0@
2batch_normalization_12_batch_normalization_12_beta:0G
9batch_normalization_12_batch_normalization_12_moving_mean:0K
=batch_normalization_12_batch_normalization_12_moving_variance:04
conv2d_13_conv2d_13_kernel:0@&
conv2d_13_conv2d_13_bias:@A
3batch_normalization_13_batch_normalization_13_gamma:@@
2batch_normalization_13_batch_normalization_13_beta:@G
9batch_normalization_13_batch_normalization_13_moving_mean:@K
=batch_normalization_13_batch_normalization_13_moving_variance:@4
conv2d_14_conv2d_14_kernel:@@&
conv2d_14_conv2d_14_bias:@A
3batch_normalization_14_batch_normalization_14_gamma:@@
2batch_normalization_14_batch_normalization_14_beta:@G
9batch_normalization_14_batch_normalization_14_moving_mean:@K
=batch_normalization_14_batch_normalization_14_moving_variance:@+
dense_6_dense_6_kernel:???#
dense_6_dense_6_bias:	?)
dense_7_dense_7_kernel:	?@"
dense_7_dense_7_bias:@(
dense_8_dense_8_kernel:@"
dense_8_dense_8_bias:
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_10_conv2d_10_kernelconv2d_10_conv2d_10_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_54952#
!conv2d_10/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:03batch_normalization_10_batch_normalization_10_gamma2batch_normalization_10_batch_normalization_10_beta9batch_normalization_10_batch_normalization_10_moving_mean=batch_normalization_10_batch_normalization_10_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_551620
.batch_normalization_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_55332#
!conv2d_11/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:03batch_normalization_11_batch_normalization_11_gamma2batch_normalization_11_batch_normalization_11_beta9batch_normalization_11_batch_normalization_11_moving_mean=batch_normalization_11_batch_normalization_11_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_555420
.batch_normalization_11/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_55712#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:03batch_normalization_12_batch_normalization_12_gamma2batch_normalization_12_batch_normalization_12_beta9batch_normalization_12_batch_normalization_12_moving_mean=batch_normalization_12_batch_normalization_12_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_559220
.batch_normalization_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_56092#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:03batch_normalization_13_batch_normalization_13_gamma2batch_normalization_13_batch_normalization_13_beta9batch_normalization_13_batch_normalization_13_moving_mean=batch_normalization_13_batch_normalization_13_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_563020
.batch_normalization_13/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_14_conv2d_14_kernelconv2d_14_conv2d_14_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_56472#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:03batch_normalization_14_batch_normalization_14_gamma2batch_normalization_14_batch_normalization_14_beta9batch_normalization_14_batch_normalization_14_moving_mean=batch_normalization_14_batch_normalization_14_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_566820
.batch_normalization_14/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_56802
flatten_2/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_dense_6_kerneldense_6_dense_6_bias*
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
GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_56932!
dense_6/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_57022
dropout_4/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_7_dense_7_kerneldense_7_dense_7_bias*
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
A__inference_dense_7_layer_call_and_return_conditional_losses_57152!
dense_7/StatefulPartitionedCall?
dropout_5/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
C__inference_dropout_5_layer_call_and_return_conditional_losses_57242
dropout_5/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_8_dense_8_kerneldense_8_dense_8_bias*
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
A__inference_dense_8_layer_call_and_return_conditional_losses_57372!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
8__inference_single_image_simple_model_layer_call_fn_6676
input_2*
conv2d_10_kernel:
conv2d_10_bias:*
batch_normalization_10_gamma:)
batch_normalization_10_beta:0
"batch_normalization_10_moving_mean:4
&batch_normalization_10_moving_variance:*
conv2d_11_kernel: 
conv2d_11_bias: *
batch_normalization_11_gamma: )
batch_normalization_11_beta: 0
"batch_normalization_11_moving_mean: 4
&batch_normalization_11_moving_variance: *
conv2d_12_kernel: 0
conv2d_12_bias:0*
batch_normalization_12_gamma:0)
batch_normalization_12_beta:00
"batch_normalization_12_moving_mean:04
&batch_normalization_12_moving_variance:0*
conv2d_13_kernel:0@
conv2d_13_bias:@*
batch_normalization_13_gamma:@)
batch_normalization_13_beta:@0
"batch_normalization_13_moving_mean:@4
&batch_normalization_13_moving_variance:@*
conv2d_14_kernel:@@
conv2d_14_bias:@*
batch_normalization_14_gamma:@)
batch_normalization_14_beta:@0
"batch_normalization_14_moving_mean:@4
&batch_normalization_14_moving_variance:@#
dense_6_kernel:???
dense_6_bias:	?!
dense_7_kernel:	?@
dense_7_bias:@ 
dense_8_kernel:@
dense_8_bias:
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_10_kernelconv2d_10_biasbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_varianceconv2d_11_kernelconv2d_11_biasbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_varianceconv2d_12_kernelconv2d_12_biasbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_varianceconv2d_13_kernelconv2d_13_biasbatch_normalization_13_gammabatch_normalization_13_beta"batch_normalization_13_moving_mean&batch_normalization_13_moving_varianceconv2d_14_kernelconv2d_14_biasbatch_normalization_14_gammabatch_normalization_14_beta"batch_normalization_14_moving_mean&batch_normalization_14_moving_variancedense_6_kerneldense_6_biasdense_7_kerneldense_7_biasdense_8_kerneldense_8_bias*0
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
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_64842
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
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7771

inputs9
+readvariableop_batch_normalization_14_gamma:@:
,readvariableop_1_batch_normalization_14_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_14_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_14_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance*
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
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_5959

inputs9
+readvariableop_batch_normalization_14_gamma:@:
,readvariableop_1_batch_normalization_14_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_14_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_14_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?
?
C__inference_conv2d_12_layer_call_and_return_conditional_losses_7465

inputs@
&conv2d_readvariableop_conv2d_12_kernel: 03
%biasadd_readvariableop_conv2d_12_bias:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_12_kernel*&
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
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_12_bias*
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
5__inference_batch_normalization_14_layer_call_fn_7735

inputs*
batch_normalization_14_gamma:@)
batch_normalization_14_beta:@0
"batch_normalization_14_moving_mean:@4
&batch_normalization_14_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_14_gammabatch_normalization_14_beta"batch_normalization_14_moving_mean&batch_normalization_14_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_53892
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
?
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_4805

inputs9
+readvariableop_batch_normalization_10_gamma::
,readvariableop_1_batch_normalization_10_beta:P
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean:V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
5__inference_batch_normalization_11_layer_call_fn_7348

inputs*
batch_normalization_11_gamma: )
batch_normalization_11_beta: 0
"batch_normalization_11_moving_mean: 4
&batch_normalization_11_moving_variance: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_49152
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
?
?
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_5243

inputs9
+readvariableop_batch_normalization_13_gamma:@:
,readvariableop_1_batch_normalization_13_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_13_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_13_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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

?
A__inference_dense_8_layer_call_and_return_conditional_losses_7944

inputs6
$matmul_readvariableop_dense_8_kernel:@1
#biasadd_readvariableop_dense_8_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_8_kernel*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_8_bias*
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
?
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7321

inputs9
+readvariableop_batch_normalization_10_gamma::
,readvariableop_1_batch_normalization_10_beta:P
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean:V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?

?
5__inference_batch_normalization_13_layer_call_fn_7609

inputs*
batch_normalization_13_gamma:@)
batch_normalization_13_beta:@0
"batch_normalization_13_moving_mean:@4
&batch_normalization_13_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_13_gammabatch_normalization_13_beta"batch_normalization_13_moving_mean&batch_normalization_13_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_52432
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
?
?
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_5207

inputs9
+readvariableop_batch_normalization_13_gamma:@:
,readvariableop_1_batch_normalization_13_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_13_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_13_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance*
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
?
?
A__inference_dense_6_layer_call_and_return_conditional_losses_5693

inputs9
$matmul_readvariableop_dense_6_kernel:???2
#biasadd_readvariableop_dense_6_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_6_kernel*!
_output_shapes
:???*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_6_bias*
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
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7789

inputs9
+readvariableop_batch_normalization_14_gamma:@:
,readvariableop_1_batch_normalization_14_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_14_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_14_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?
?
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7645

inputs9
+readvariableop_batch_normalization_13_gamma:@:
,readvariableop_1_batch_normalization_13_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_13_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_13_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance*
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
?

?
5__inference_batch_normalization_14_layer_call_fn_7744

inputs*
batch_normalization_14_gamma:@)
batch_normalization_14_beta:@0
"batch_normalization_14_moving_mean:@4
&batch_normalization_14_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_14_gammabatch_normalization_14_beta"batch_normalization_14_moving_mean&batch_normalization_14_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_56682
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
?
?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7393

inputs9
+readvariableop_batch_normalization_11_gamma: :
,readvariableop_1_batch_normalization_11_beta: P
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean: V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
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
 
_user_specified_nameinputs
?

?
5__inference_batch_normalization_13_layer_call_fn_7618

inputs*
batch_normalization_13_gamma:@)
batch_normalization_13_beta:@0
"batch_normalization_13_moving_mean:@4
&batch_normalization_13_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_13_gammabatch_normalization_13_beta"batch_normalization_13_moving_mean&batch_normalization_13_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_56302
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
?
?
8__inference_single_image_simple_model_layer_call_fn_5781
input_2*
conv2d_10_kernel:
conv2d_10_bias:*
batch_normalization_10_gamma:)
batch_normalization_10_beta:0
"batch_normalization_10_moving_mean:4
&batch_normalization_10_moving_variance:*
conv2d_11_kernel: 
conv2d_11_bias: *
batch_normalization_11_gamma: )
batch_normalization_11_beta: 0
"batch_normalization_11_moving_mean: 4
&batch_normalization_11_moving_variance: *
conv2d_12_kernel: 0
conv2d_12_bias:0*
batch_normalization_12_gamma:0)
batch_normalization_12_beta:00
"batch_normalization_12_moving_mean:04
&batch_normalization_12_moving_variance:0*
conv2d_13_kernel:0@
conv2d_13_bias:@*
batch_normalization_13_gamma:@)
batch_normalization_13_beta:@0
"batch_normalization_13_moving_mean:@4
&batch_normalization_13_moving_variance:@*
conv2d_14_kernel:@@
conv2d_14_bias:@*
batch_normalization_14_gamma:@)
batch_normalization_14_beta:@0
"batch_normalization_14_moving_mean:@4
&batch_normalization_14_moving_variance:@#
dense_6_kernel:???
dense_6_bias:	?!
dense_7_kernel:	?@
dense_7_bias:@ 
dense_8_kernel:@
dense_8_bias:
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_10_kernelconv2d_10_biasbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_varianceconv2d_11_kernelconv2d_11_biasbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_varianceconv2d_12_kernelconv2d_12_biasbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_varianceconv2d_13_kernelconv2d_13_biasbatch_normalization_13_gammabatch_normalization_13_beta"batch_normalization_13_moving_mean&batch_normalization_13_moving_varianceconv2d_14_kernelconv2d_14_biasbatch_normalization_14_gammabatch_normalization_14_beta"batch_normalization_14_moving_mean&batch_normalization_14_moving_variancedense_6_kerneldense_6_biasdense_7_kerneldense_7_biasdense_8_kerneldense_8_bias*0
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
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_57422
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
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
a
C__inference_dropout_4_layer_call_and_return_conditional_losses_5702

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
?
a
C__inference_dropout_4_layer_call_and_return_conditional_losses_7869

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
??
?2
__inference__wrapped_model_4747
input_2d
Jsingle_image_simple_model_conv2d_10_conv2d_readvariableop_conv2d_10_kernel:W
Isingle_image_simple_model_conv2d_10_biasadd_readvariableop_conv2d_10_bias:j
\single_image_simple_model_batch_normalization_10_readvariableop_batch_normalization_10_gamma:k
]single_image_simple_model_batch_normalization_10_readvariableop_1_batch_normalization_10_beta:?
ssingle_image_simple_model_batch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean:?
ysingle_image_simple_model_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance:d
Jsingle_image_simple_model_conv2d_11_conv2d_readvariableop_conv2d_11_kernel: W
Isingle_image_simple_model_conv2d_11_biasadd_readvariableop_conv2d_11_bias: j
\single_image_simple_model_batch_normalization_11_readvariableop_batch_normalization_11_gamma: k
]single_image_simple_model_batch_normalization_11_readvariableop_1_batch_normalization_11_beta: ?
ssingle_image_simple_model_batch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean: ?
ysingle_image_simple_model_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance: d
Jsingle_image_simple_model_conv2d_12_conv2d_readvariableop_conv2d_12_kernel: 0W
Isingle_image_simple_model_conv2d_12_biasadd_readvariableop_conv2d_12_bias:0j
\single_image_simple_model_batch_normalization_12_readvariableop_batch_normalization_12_gamma:0k
]single_image_simple_model_batch_normalization_12_readvariableop_1_batch_normalization_12_beta:0?
ssingle_image_simple_model_batch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean:0?
ysingle_image_simple_model_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance:0d
Jsingle_image_simple_model_conv2d_13_conv2d_readvariableop_conv2d_13_kernel:0@W
Isingle_image_simple_model_conv2d_13_biasadd_readvariableop_conv2d_13_bias:@j
\single_image_simple_model_batch_normalization_13_readvariableop_batch_normalization_13_gamma:@k
]single_image_simple_model_batch_normalization_13_readvariableop_1_batch_normalization_13_beta:@?
ssingle_image_simple_model_batch_normalization_13_fusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean:@?
ysingle_image_simple_model_batch_normalization_13_fusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance:@d
Jsingle_image_simple_model_conv2d_14_conv2d_readvariableop_conv2d_14_kernel:@@W
Isingle_image_simple_model_conv2d_14_biasadd_readvariableop_conv2d_14_bias:@j
\single_image_simple_model_batch_normalization_14_readvariableop_batch_normalization_14_gamma:@k
]single_image_simple_model_batch_normalization_14_readvariableop_1_batch_normalization_14_beta:@?
ssingle_image_simple_model_batch_normalization_14_fusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean:@?
ysingle_image_simple_model_batch_normalization_14_fusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance:@[
Fsingle_image_simple_model_dense_6_matmul_readvariableop_dense_6_kernel:???T
Esingle_image_simple_model_dense_6_biasadd_readvariableop_dense_6_bias:	?Y
Fsingle_image_simple_model_dense_7_matmul_readvariableop_dense_7_kernel:	?@S
Esingle_image_simple_model_dense_7_biasadd_readvariableop_dense_7_bias:@X
Fsingle_image_simple_model_dense_8_matmul_readvariableop_dense_8_kernel:@S
Esingle_image_simple_model_dense_8_biasadd_readvariableop_dense_8_bias:
identity??Psingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?Rsingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1??single_image_simple_model/batch_normalization_10/ReadVariableOp?Asingle_image_simple_model/batch_normalization_10/ReadVariableOp_1?Psingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?Rsingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1??single_image_simple_model/batch_normalization_11/ReadVariableOp?Asingle_image_simple_model/batch_normalization_11/ReadVariableOp_1?Psingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?Rsingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1??single_image_simple_model/batch_normalization_12/ReadVariableOp?Asingle_image_simple_model/batch_normalization_12/ReadVariableOp_1?Psingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?Rsingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1??single_image_simple_model/batch_normalization_13/ReadVariableOp?Asingle_image_simple_model/batch_normalization_13/ReadVariableOp_1?Psingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?Rsingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1??single_image_simple_model/batch_normalization_14/ReadVariableOp?Asingle_image_simple_model/batch_normalization_14/ReadVariableOp_1?:single_image_simple_model/conv2d_10/BiasAdd/ReadVariableOp?9single_image_simple_model/conv2d_10/Conv2D/ReadVariableOp?:single_image_simple_model/conv2d_11/BiasAdd/ReadVariableOp?9single_image_simple_model/conv2d_11/Conv2D/ReadVariableOp?:single_image_simple_model/conv2d_12/BiasAdd/ReadVariableOp?9single_image_simple_model/conv2d_12/Conv2D/ReadVariableOp?:single_image_simple_model/conv2d_13/BiasAdd/ReadVariableOp?9single_image_simple_model/conv2d_13/Conv2D/ReadVariableOp?:single_image_simple_model/conv2d_14/BiasAdd/ReadVariableOp?9single_image_simple_model/conv2d_14/Conv2D/ReadVariableOp?8single_image_simple_model/dense_6/BiasAdd/ReadVariableOp?7single_image_simple_model/dense_6/MatMul/ReadVariableOp?8single_image_simple_model/dense_7/BiasAdd/ReadVariableOp?7single_image_simple_model/dense_7/MatMul/ReadVariableOp?8single_image_simple_model/dense_8/BiasAdd/ReadVariableOp?7single_image_simple_model/dense_8/MatMul/ReadVariableOp?
9single_image_simple_model/conv2d_10/Conv2D/ReadVariableOpReadVariableOpJsingle_image_simple_model_conv2d_10_conv2d_readvariableop_conv2d_10_kernel*&
_output_shapes
:*
dtype02;
9single_image_simple_model/conv2d_10/Conv2D/ReadVariableOp?
*single_image_simple_model/conv2d_10/Conv2DConv2Dinput_2Asingle_image_simple_model/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2,
*single_image_simple_model/conv2d_10/Conv2D?
:single_image_simple_model/conv2d_10/BiasAdd/ReadVariableOpReadVariableOpIsingle_image_simple_model_conv2d_10_biasadd_readvariableop_conv2d_10_bias*
_output_shapes
:*
dtype02<
:single_image_simple_model/conv2d_10/BiasAdd/ReadVariableOp?
+single_image_simple_model/conv2d_10/BiasAddBiasAdd3single_image_simple_model/conv2d_10/Conv2D:output:0Bsingle_image_simple_model/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2-
+single_image_simple_model/conv2d_10/BiasAdd?
(single_image_simple_model/conv2d_10/ReluRelu4single_image_simple_model/conv2d_10/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2*
(single_image_simple_model/conv2d_10/Relu?
?single_image_simple_model/batch_normalization_10/ReadVariableOpReadVariableOp\single_image_simple_model_batch_normalization_10_readvariableop_batch_normalization_10_gamma*
_output_shapes
:*
dtype02A
?single_image_simple_model/batch_normalization_10/ReadVariableOp?
Asingle_image_simple_model/batch_normalization_10/ReadVariableOp_1ReadVariableOp]single_image_simple_model_batch_normalization_10_readvariableop_1_batch_normalization_10_beta*
_output_shapes
:*
dtype02C
Asingle_image_simple_model/batch_normalization_10/ReadVariableOp_1?
Psingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpssingle_image_simple_model_batch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:*
dtype02R
Psingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
Rsingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpysingle_image_simple_model_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:*
dtype02T
Rsingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
Asingle_image_simple_model/batch_normalization_10/FusedBatchNormV3FusedBatchNormV36single_image_simple_model/conv2d_10/Relu:activations:0Gsingle_image_simple_model/batch_normalization_10/ReadVariableOp:value:0Isingle_image_simple_model/batch_normalization_10/ReadVariableOp_1:value:0Xsingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Zsingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2C
Asingle_image_simple_model/batch_normalization_10/FusedBatchNormV3?
9single_image_simple_model/conv2d_11/Conv2D/ReadVariableOpReadVariableOpJsingle_image_simple_model_conv2d_11_conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
: *
dtype02;
9single_image_simple_model/conv2d_11/Conv2D/ReadVariableOp?
*single_image_simple_model/conv2d_11/Conv2DConv2DEsingle_image_simple_model/batch_normalization_10/FusedBatchNormV3:y:0Asingle_image_simple_model/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH *
paddingVALID*
strides
2,
*single_image_simple_model/conv2d_11/Conv2D?
:single_image_simple_model/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpIsingle_image_simple_model_conv2d_11_biasadd_readvariableop_conv2d_11_bias*
_output_shapes
: *
dtype02<
:single_image_simple_model/conv2d_11/BiasAdd/ReadVariableOp?
+single_image_simple_model/conv2d_11/BiasAddBiasAdd3single_image_simple_model/conv2d_11/Conv2D:output:0Bsingle_image_simple_model/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH 2-
+single_image_simple_model/conv2d_11/BiasAdd?
(single_image_simple_model/conv2d_11/ReluRelu4single_image_simple_model/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????HH 2*
(single_image_simple_model/conv2d_11/Relu?
?single_image_simple_model/batch_normalization_11/ReadVariableOpReadVariableOp\single_image_simple_model_batch_normalization_11_readvariableop_batch_normalization_11_gamma*
_output_shapes
: *
dtype02A
?single_image_simple_model/batch_normalization_11/ReadVariableOp?
Asingle_image_simple_model/batch_normalization_11/ReadVariableOp_1ReadVariableOp]single_image_simple_model_batch_normalization_11_readvariableop_1_batch_normalization_11_beta*
_output_shapes
: *
dtype02C
Asingle_image_simple_model/batch_normalization_11/ReadVariableOp_1?
Psingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpssingle_image_simple_model_batch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
: *
dtype02R
Psingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
Rsingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpysingle_image_simple_model_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
: *
dtype02T
Rsingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
Asingle_image_simple_model/batch_normalization_11/FusedBatchNormV3FusedBatchNormV36single_image_simple_model/conv2d_11/Relu:activations:0Gsingle_image_simple_model/batch_normalization_11/ReadVariableOp:value:0Isingle_image_simple_model/batch_normalization_11/ReadVariableOp_1:value:0Xsingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Zsingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????HH : : : : :*
epsilon%o?:*
is_training( 2C
Asingle_image_simple_model/batch_normalization_11/FusedBatchNormV3?
9single_image_simple_model/conv2d_12/Conv2D/ReadVariableOpReadVariableOpJsingle_image_simple_model_conv2d_12_conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: 0*
dtype02;
9single_image_simple_model/conv2d_12/Conv2D/ReadVariableOp?
*single_image_simple_model/conv2d_12/Conv2DConv2DEsingle_image_simple_model/batch_normalization_11/FusedBatchNormV3:y:0Asingle_image_simple_model/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##0*
paddingVALID*
strides
2,
*single_image_simple_model/conv2d_12/Conv2D?
:single_image_simple_model/conv2d_12/BiasAdd/ReadVariableOpReadVariableOpIsingle_image_simple_model_conv2d_12_biasadd_readvariableop_conv2d_12_bias*
_output_shapes
:0*
dtype02<
:single_image_simple_model/conv2d_12/BiasAdd/ReadVariableOp?
+single_image_simple_model/conv2d_12/BiasAddBiasAdd3single_image_simple_model/conv2d_12/Conv2D:output:0Bsingle_image_simple_model/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##02-
+single_image_simple_model/conv2d_12/BiasAdd?
(single_image_simple_model/conv2d_12/ReluRelu4single_image_simple_model/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????##02*
(single_image_simple_model/conv2d_12/Relu?
?single_image_simple_model/batch_normalization_12/ReadVariableOpReadVariableOp\single_image_simple_model_batch_normalization_12_readvariableop_batch_normalization_12_gamma*
_output_shapes
:0*
dtype02A
?single_image_simple_model/batch_normalization_12/ReadVariableOp?
Asingle_image_simple_model/batch_normalization_12/ReadVariableOp_1ReadVariableOp]single_image_simple_model_batch_normalization_12_readvariableop_1_batch_normalization_12_beta*
_output_shapes
:0*
dtype02C
Asingle_image_simple_model/batch_normalization_12/ReadVariableOp_1?
Psingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpssingle_image_simple_model_batch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:0*
dtype02R
Psingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
Rsingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpysingle_image_simple_model_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:0*
dtype02T
Rsingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
Asingle_image_simple_model/batch_normalization_12/FusedBatchNormV3FusedBatchNormV36single_image_simple_model/conv2d_12/Relu:activations:0Gsingle_image_simple_model/batch_normalization_12/ReadVariableOp:value:0Isingle_image_simple_model/batch_normalization_12/ReadVariableOp_1:value:0Xsingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Zsingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????##0:0:0:0:0:*
epsilon%o?:*
is_training( 2C
Asingle_image_simple_model/batch_normalization_12/FusedBatchNormV3?
9single_image_simple_model/conv2d_13/Conv2D/ReadVariableOpReadVariableOpJsingle_image_simple_model_conv2d_13_conv2d_readvariableop_conv2d_13_kernel*&
_output_shapes
:0@*
dtype02;
9single_image_simple_model/conv2d_13/Conv2D/ReadVariableOp?
*single_image_simple_model/conv2d_13/Conv2DConv2DEsingle_image_simple_model/batch_normalization_12/FusedBatchNormV3:y:0Asingle_image_simple_model/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@*
paddingVALID*
strides
2,
*single_image_simple_model/conv2d_13/Conv2D?
:single_image_simple_model/conv2d_13/BiasAdd/ReadVariableOpReadVariableOpIsingle_image_simple_model_conv2d_13_biasadd_readvariableop_conv2d_13_bias*
_output_shapes
:@*
dtype02<
:single_image_simple_model/conv2d_13/BiasAdd/ReadVariableOp?
+single_image_simple_model/conv2d_13/BiasAddBiasAdd3single_image_simple_model/conv2d_13/Conv2D:output:0Bsingle_image_simple_model/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@2-
+single_image_simple_model/conv2d_13/BiasAdd?
(single_image_simple_model/conv2d_13/ReluRelu4single_image_simple_model/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!!@2*
(single_image_simple_model/conv2d_13/Relu?
?single_image_simple_model/batch_normalization_13/ReadVariableOpReadVariableOp\single_image_simple_model_batch_normalization_13_readvariableop_batch_normalization_13_gamma*
_output_shapes
:@*
dtype02A
?single_image_simple_model/batch_normalization_13/ReadVariableOp?
Asingle_image_simple_model/batch_normalization_13/ReadVariableOp_1ReadVariableOp]single_image_simple_model_batch_normalization_13_readvariableop_1_batch_normalization_13_beta*
_output_shapes
:@*
dtype02C
Asingle_image_simple_model/batch_normalization_13/ReadVariableOp_1?
Psingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpssingle_image_simple_model_batch_normalization_13_fusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean*
_output_shapes
:@*
dtype02R
Psingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
Rsingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpysingle_image_simple_model_batch_normalization_13_fusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance*
_output_shapes
:@*
dtype02T
Rsingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
Asingle_image_simple_model/batch_normalization_13/FusedBatchNormV3FusedBatchNormV36single_image_simple_model/conv2d_13/Relu:activations:0Gsingle_image_simple_model/batch_normalization_13/ReadVariableOp:value:0Isingle_image_simple_model/batch_normalization_13/ReadVariableOp_1:value:0Xsingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Zsingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????!!@:@:@:@:@:*
epsilon%o?:*
is_training( 2C
Asingle_image_simple_model/batch_normalization_13/FusedBatchNormV3?
9single_image_simple_model/conv2d_14/Conv2D/ReadVariableOpReadVariableOpJsingle_image_simple_model_conv2d_14_conv2d_readvariableop_conv2d_14_kernel*&
_output_shapes
:@@*
dtype02;
9single_image_simple_model/conv2d_14/Conv2D/ReadVariableOp?
*single_image_simple_model/conv2d_14/Conv2DConv2DEsingle_image_simple_model/batch_normalization_13/FusedBatchNormV3:y:0Asingle_image_simple_model/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2,
*single_image_simple_model/conv2d_14/Conv2D?
:single_image_simple_model/conv2d_14/BiasAdd/ReadVariableOpReadVariableOpIsingle_image_simple_model_conv2d_14_biasadd_readvariableop_conv2d_14_bias*
_output_shapes
:@*
dtype02<
:single_image_simple_model/conv2d_14/BiasAdd/ReadVariableOp?
+single_image_simple_model/conv2d_14/BiasAddBiasAdd3single_image_simple_model/conv2d_14/Conv2D:output:0Bsingle_image_simple_model/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2-
+single_image_simple_model/conv2d_14/BiasAdd?
(single_image_simple_model/conv2d_14/ReluRelu4single_image_simple_model/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2*
(single_image_simple_model/conv2d_14/Relu?
?single_image_simple_model/batch_normalization_14/ReadVariableOpReadVariableOp\single_image_simple_model_batch_normalization_14_readvariableop_batch_normalization_14_gamma*
_output_shapes
:@*
dtype02A
?single_image_simple_model/batch_normalization_14/ReadVariableOp?
Asingle_image_simple_model/batch_normalization_14/ReadVariableOp_1ReadVariableOp]single_image_simple_model_batch_normalization_14_readvariableop_1_batch_normalization_14_beta*
_output_shapes
:@*
dtype02C
Asingle_image_simple_model/batch_normalization_14/ReadVariableOp_1?
Psingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpssingle_image_simple_model_batch_normalization_14_fusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean*
_output_shapes
:@*
dtype02R
Psingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
Rsingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpysingle_image_simple_model_batch_normalization_14_fusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance*
_output_shapes
:@*
dtype02T
Rsingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
Asingle_image_simple_model/batch_normalization_14/FusedBatchNormV3FusedBatchNormV36single_image_simple_model/conv2d_14/Relu:activations:0Gsingle_image_simple_model/batch_normalization_14/ReadVariableOp:value:0Isingle_image_simple_model/batch_normalization_14/ReadVariableOp_1:value:0Xsingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Zsingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2C
Asingle_image_simple_model/batch_normalization_14/FusedBatchNormV3?
)single_image_simple_model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@?  2+
)single_image_simple_model/flatten_2/Const?
+single_image_simple_model/flatten_2/ReshapeReshapeEsingle_image_simple_model/batch_normalization_14/FusedBatchNormV3:y:02single_image_simple_model/flatten_2/Const:output:0*
T0*)
_output_shapes
:???????????2-
+single_image_simple_model/flatten_2/Reshape?
7single_image_simple_model/dense_6/MatMul/ReadVariableOpReadVariableOpFsingle_image_simple_model_dense_6_matmul_readvariableop_dense_6_kernel*!
_output_shapes
:???*
dtype029
7single_image_simple_model/dense_6/MatMul/ReadVariableOp?
(single_image_simple_model/dense_6/MatMulMatMul4single_image_simple_model/flatten_2/Reshape:output:0?single_image_simple_model/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(single_image_simple_model/dense_6/MatMul?
8single_image_simple_model/dense_6/BiasAdd/ReadVariableOpReadVariableOpEsingle_image_simple_model_dense_6_biasadd_readvariableop_dense_6_bias*
_output_shapes	
:?*
dtype02:
8single_image_simple_model/dense_6/BiasAdd/ReadVariableOp?
)single_image_simple_model/dense_6/BiasAddBiasAdd2single_image_simple_model/dense_6/MatMul:product:0@single_image_simple_model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)single_image_simple_model/dense_6/BiasAdd?
&single_image_simple_model/dense_6/ReluRelu2single_image_simple_model/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&single_image_simple_model/dense_6/Relu?
,single_image_simple_model/dropout_4/IdentityIdentity4single_image_simple_model/dense_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,single_image_simple_model/dropout_4/Identity?
7single_image_simple_model/dense_7/MatMul/ReadVariableOpReadVariableOpFsingle_image_simple_model_dense_7_matmul_readvariableop_dense_7_kernel*
_output_shapes
:	?@*
dtype029
7single_image_simple_model/dense_7/MatMul/ReadVariableOp?
(single_image_simple_model/dense_7/MatMulMatMul5single_image_simple_model/dropout_4/Identity:output:0?single_image_simple_model/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2*
(single_image_simple_model/dense_7/MatMul?
8single_image_simple_model/dense_7/BiasAdd/ReadVariableOpReadVariableOpEsingle_image_simple_model_dense_7_biasadd_readvariableop_dense_7_bias*
_output_shapes
:@*
dtype02:
8single_image_simple_model/dense_7/BiasAdd/ReadVariableOp?
)single_image_simple_model/dense_7/BiasAddBiasAdd2single_image_simple_model/dense_7/MatMul:product:0@single_image_simple_model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2+
)single_image_simple_model/dense_7/BiasAdd?
&single_image_simple_model/dense_7/ReluRelu2single_image_simple_model/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2(
&single_image_simple_model/dense_7/Relu?
,single_image_simple_model/dropout_5/IdentityIdentity4single_image_simple_model/dense_7/Relu:activations:0*
T0*'
_output_shapes
:?????????@2.
,single_image_simple_model/dropout_5/Identity?
7single_image_simple_model/dense_8/MatMul/ReadVariableOpReadVariableOpFsingle_image_simple_model_dense_8_matmul_readvariableop_dense_8_kernel*
_output_shapes

:@*
dtype029
7single_image_simple_model/dense_8/MatMul/ReadVariableOp?
(single_image_simple_model/dense_8/MatMulMatMul5single_image_simple_model/dropout_5/Identity:output:0?single_image_simple_model/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(single_image_simple_model/dense_8/MatMul?
8single_image_simple_model/dense_8/BiasAdd/ReadVariableOpReadVariableOpEsingle_image_simple_model_dense_8_biasadd_readvariableop_dense_8_bias*
_output_shapes
:*
dtype02:
8single_image_simple_model/dense_8/BiasAdd/ReadVariableOp?
)single_image_simple_model/dense_8/BiasAddBiasAdd2single_image_simple_model/dense_8/MatMul:product:0@single_image_simple_model/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)single_image_simple_model/dense_8/BiasAdd?
)single_image_simple_model/dense_8/SigmoidSigmoid2single_image_simple_model/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2+
)single_image_simple_model/dense_8/Sigmoid?
IdentityIdentity-single_image_simple_model/dense_8/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOpQ^single_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOpS^single_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1@^single_image_simple_model/batch_normalization_10/ReadVariableOpB^single_image_simple_model/batch_normalization_10/ReadVariableOp_1Q^single_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOpS^single_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1@^single_image_simple_model/batch_normalization_11/ReadVariableOpB^single_image_simple_model/batch_normalization_11/ReadVariableOp_1Q^single_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOpS^single_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1@^single_image_simple_model/batch_normalization_12/ReadVariableOpB^single_image_simple_model/batch_normalization_12/ReadVariableOp_1Q^single_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOpS^single_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1@^single_image_simple_model/batch_normalization_13/ReadVariableOpB^single_image_simple_model/batch_normalization_13/ReadVariableOp_1Q^single_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOpS^single_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1@^single_image_simple_model/batch_normalization_14/ReadVariableOpB^single_image_simple_model/batch_normalization_14/ReadVariableOp_1;^single_image_simple_model/conv2d_10/BiasAdd/ReadVariableOp:^single_image_simple_model/conv2d_10/Conv2D/ReadVariableOp;^single_image_simple_model/conv2d_11/BiasAdd/ReadVariableOp:^single_image_simple_model/conv2d_11/Conv2D/ReadVariableOp;^single_image_simple_model/conv2d_12/BiasAdd/ReadVariableOp:^single_image_simple_model/conv2d_12/Conv2D/ReadVariableOp;^single_image_simple_model/conv2d_13/BiasAdd/ReadVariableOp:^single_image_simple_model/conv2d_13/Conv2D/ReadVariableOp;^single_image_simple_model/conv2d_14/BiasAdd/ReadVariableOp:^single_image_simple_model/conv2d_14/Conv2D/ReadVariableOp9^single_image_simple_model/dense_6/BiasAdd/ReadVariableOp8^single_image_simple_model/dense_6/MatMul/ReadVariableOp9^single_image_simple_model/dense_7/BiasAdd/ReadVariableOp8^single_image_simple_model/dense_7/MatMul/ReadVariableOp9^single_image_simple_model/dense_8/BiasAdd/ReadVariableOp8^single_image_simple_model/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Psingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOpPsingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2?
Rsingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Rsingle_image_simple_model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12?
?single_image_simple_model/batch_normalization_10/ReadVariableOp?single_image_simple_model/batch_normalization_10/ReadVariableOp2?
Asingle_image_simple_model/batch_normalization_10/ReadVariableOp_1Asingle_image_simple_model/batch_normalization_10/ReadVariableOp_12?
Psingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOpPsingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2?
Rsingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Rsingle_image_simple_model/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12?
?single_image_simple_model/batch_normalization_11/ReadVariableOp?single_image_simple_model/batch_normalization_11/ReadVariableOp2?
Asingle_image_simple_model/batch_normalization_11/ReadVariableOp_1Asingle_image_simple_model/batch_normalization_11/ReadVariableOp_12?
Psingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOpPsingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
Rsingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Rsingle_image_simple_model/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12?
?single_image_simple_model/batch_normalization_12/ReadVariableOp?single_image_simple_model/batch_normalization_12/ReadVariableOp2?
Asingle_image_simple_model/batch_normalization_12/ReadVariableOp_1Asingle_image_simple_model/batch_normalization_12/ReadVariableOp_12?
Psingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOpPsingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
Rsingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Rsingle_image_simple_model/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12?
?single_image_simple_model/batch_normalization_13/ReadVariableOp?single_image_simple_model/batch_normalization_13/ReadVariableOp2?
Asingle_image_simple_model/batch_normalization_13/ReadVariableOp_1Asingle_image_simple_model/batch_normalization_13/ReadVariableOp_12?
Psingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOpPsingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2?
Rsingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Rsingle_image_simple_model/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12?
?single_image_simple_model/batch_normalization_14/ReadVariableOp?single_image_simple_model/batch_normalization_14/ReadVariableOp2?
Asingle_image_simple_model/batch_normalization_14/ReadVariableOp_1Asingle_image_simple_model/batch_normalization_14/ReadVariableOp_12x
:single_image_simple_model/conv2d_10/BiasAdd/ReadVariableOp:single_image_simple_model/conv2d_10/BiasAdd/ReadVariableOp2v
9single_image_simple_model/conv2d_10/Conv2D/ReadVariableOp9single_image_simple_model/conv2d_10/Conv2D/ReadVariableOp2x
:single_image_simple_model/conv2d_11/BiasAdd/ReadVariableOp:single_image_simple_model/conv2d_11/BiasAdd/ReadVariableOp2v
9single_image_simple_model/conv2d_11/Conv2D/ReadVariableOp9single_image_simple_model/conv2d_11/Conv2D/ReadVariableOp2x
:single_image_simple_model/conv2d_12/BiasAdd/ReadVariableOp:single_image_simple_model/conv2d_12/BiasAdd/ReadVariableOp2v
9single_image_simple_model/conv2d_12/Conv2D/ReadVariableOp9single_image_simple_model/conv2d_12/Conv2D/ReadVariableOp2x
:single_image_simple_model/conv2d_13/BiasAdd/ReadVariableOp:single_image_simple_model/conv2d_13/BiasAdd/ReadVariableOp2v
9single_image_simple_model/conv2d_13/Conv2D/ReadVariableOp9single_image_simple_model/conv2d_13/Conv2D/ReadVariableOp2x
:single_image_simple_model/conv2d_14/BiasAdd/ReadVariableOp:single_image_simple_model/conv2d_14/BiasAdd/ReadVariableOp2v
9single_image_simple_model/conv2d_14/Conv2D/ReadVariableOp9single_image_simple_model/conv2d_14/Conv2D/ReadVariableOp2t
8single_image_simple_model/dense_6/BiasAdd/ReadVariableOp8single_image_simple_model/dense_6/BiasAdd/ReadVariableOp2r
7single_image_simple_model/dense_6/MatMul/ReadVariableOp7single_image_simple_model/dense_6/MatMul/ReadVariableOp2t
8single_image_simple_model/dense_7/BiasAdd/ReadVariableOp8single_image_simple_model/dense_7/BiasAdd/ReadVariableOp2r
7single_image_simple_model/dense_7/MatMul/ReadVariableOp7single_image_simple_model/dense_7/MatMul/ReadVariableOp2t
8single_image_simple_model/dense_8/BiasAdd/ReadVariableOp8single_image_simple_model/dense_8/BiasAdd/ReadVariableOp2r
7single_image_simple_model/dense_8/MatMul/ReadVariableOp7single_image_simple_model/dense_8/MatMul/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_5680

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
?
?
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_5353

inputs9
+readvariableop_batch_normalization_14_gamma:@:
,readvariableop_1_batch_normalization_14_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_14_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_14_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance*
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
??
?$
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_7047

inputsJ
0conv2d_10_conv2d_readvariableop_conv2d_10_kernel:=
/conv2d_10_biasadd_readvariableop_conv2d_10_bias:P
Bbatch_normalization_10_readvariableop_batch_normalization_10_gamma:Q
Cbatch_normalization_10_readvariableop_1_batch_normalization_10_beta:g
Ybatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean:m
_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance:J
0conv2d_11_conv2d_readvariableop_conv2d_11_kernel: =
/conv2d_11_biasadd_readvariableop_conv2d_11_bias: P
Bbatch_normalization_11_readvariableop_batch_normalization_11_gamma: Q
Cbatch_normalization_11_readvariableop_1_batch_normalization_11_beta: g
Ybatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean: m
_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance: J
0conv2d_12_conv2d_readvariableop_conv2d_12_kernel: 0=
/conv2d_12_biasadd_readvariableop_conv2d_12_bias:0P
Bbatch_normalization_12_readvariableop_batch_normalization_12_gamma:0Q
Cbatch_normalization_12_readvariableop_1_batch_normalization_12_beta:0g
Ybatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean:0m
_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance:0J
0conv2d_13_conv2d_readvariableop_conv2d_13_kernel:0@=
/conv2d_13_biasadd_readvariableop_conv2d_13_bias:@P
Bbatch_normalization_13_readvariableop_batch_normalization_13_gamma:@Q
Cbatch_normalization_13_readvariableop_1_batch_normalization_13_beta:@g
Ybatch_normalization_13_fusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean:@m
_batch_normalization_13_fusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance:@J
0conv2d_14_conv2d_readvariableop_conv2d_14_kernel:@@=
/conv2d_14_biasadd_readvariableop_conv2d_14_bias:@P
Bbatch_normalization_14_readvariableop_batch_normalization_14_gamma:@Q
Cbatch_normalization_14_readvariableop_1_batch_normalization_14_beta:@g
Ybatch_normalization_14_fusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean:@m
_batch_normalization_14_fusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance:@A
,dense_6_matmul_readvariableop_dense_6_kernel:???:
+dense_6_biasadd_readvariableop_dense_6_bias:	??
,dense_7_matmul_readvariableop_dense_7_kernel:	?@9
+dense_7_biasadd_readvariableop_dense_7_bias:@>
,dense_8_matmul_readvariableop_dense_8_kernel:@9
+dense_8_biasadd_readvariableop_dense_8_bias:
identity??6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_11/ReadVariableOp?'batch_normalization_11/ReadVariableOp_1?6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp0conv2d_10_conv2d_readvariableop_conv2d_10_kernel*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp/conv2d_10_biasadd_readvariableop_conv2d_10_bias*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_10/BiasAdd?
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_10/Relu?
%batch_normalization_10/ReadVariableOpReadVariableOpBbatch_normalization_10_readvariableop_batch_normalization_10_gamma*
_output_shapes
:*
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOpCbatch_normalization_10_readvariableop_1_batch_normalization_10_beta*
_output_shapes
:*
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp0conv2d_11_conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
: *
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH *
paddingVALID*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp/conv2d_11_biasadd_readvariableop_conv2d_11_bias*
_output_shapes
: *
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????HH 2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????HH 2
conv2d_11/Relu?
%batch_normalization_11/ReadVariableOpReadVariableOpBbatch_normalization_11_readvariableop_batch_normalization_11_gamma*
_output_shapes
: *
dtype02'
%batch_normalization_11/ReadVariableOp?
'batch_normalization_11/ReadVariableOp_1ReadVariableOpCbatch_normalization_11_readvariableop_1_batch_normalization_11_beta*
_output_shapes
: *
dtype02)
'batch_normalization_11/ReadVariableOp_1?
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
: *
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
: *
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????HH : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp0conv2d_12_conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: 0*
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D+batch_normalization_11/FusedBatchNormV3:y:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##0*
paddingVALID*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp/conv2d_12_biasadd_readvariableop_conv2d_12_bias*
_output_shapes
:0*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????##02
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????##02
conv2d_12/Relu?
%batch_normalization_12/ReadVariableOpReadVariableOpBbatch_normalization_12_readvariableop_batch_normalization_12_gamma*
_output_shapes
:0*
dtype02'
%batch_normalization_12/ReadVariableOp?
'batch_normalization_12/ReadVariableOp_1ReadVariableOpCbatch_normalization_12_readvariableop_1_batch_normalization_12_beta*
_output_shapes
:0*
dtype02)
'batch_normalization_12/ReadVariableOp_1?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:0*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:0*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_12/Relu:activations:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????##0:0:0:0:0:*
epsilon%o?:*
is_training( 2)
'batch_normalization_12/FusedBatchNormV3?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp0conv2d_13_conv2d_readvariableop_conv2d_13_kernel*&
_output_shapes
:0@*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D+batch_normalization_12/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@*
paddingVALID*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp/conv2d_13_biasadd_readvariableop_conv2d_13_bias*
_output_shapes
:@*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!!@2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!!@2
conv2d_13/Relu?
%batch_normalization_13/ReadVariableOpReadVariableOpBbatch_normalization_13_readvariableop_batch_normalization_13_gamma*
_output_shapes
:@*
dtype02'
%batch_normalization_13/ReadVariableOp?
'batch_normalization_13/ReadVariableOp_1ReadVariableOpCbatch_normalization_13_readvariableop_1_batch_normalization_13_beta*
_output_shapes
:@*
dtype02)
'batch_normalization_13/ReadVariableOp_1?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_13_fusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean*
_output_shapes
:@*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_13_fusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance*
_output_shapes
:@*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????!!@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_13/FusedBatchNormV3?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp0conv2d_14_conv2d_readvariableop_conv2d_14_kernel*&
_output_shapes
:@@*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D+batch_normalization_13/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp/conv2d_14_biasadd_readvariableop_conv2d_14_bias*
_output_shapes
:@*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/BiasAdd~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/Relu?
%batch_normalization_14/ReadVariableOpReadVariableOpBbatch_normalization_14_readvariableop_batch_normalization_14_gamma*
_output_shapes
:@*
dtype02'
%batch_normalization_14/ReadVariableOp?
'batch_normalization_14/ReadVariableOp_1ReadVariableOpCbatch_normalization_14_readvariableop_1_batch_normalization_14_beta*
_output_shapes
:@*
dtype02)
'batch_normalization_14/ReadVariableOp_1?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_14_fusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean*
_output_shapes
:@*
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_14_fusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance*
_output_shapes
:@*
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_14/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_14/FusedBatchNormV3s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@?  2
flatten_2/Const?
flatten_2/ReshapeReshape+batch_normalization_14/FusedBatchNormV3:y:0flatten_2/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_2/Reshape?
dense_6/MatMul/ReadVariableOpReadVariableOp,dense_6_matmul_readvariableop_dense_6_kernel*!
_output_shapes
:???*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp+dense_6_biasadd_readvariableop_dense_6_bias*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Relu?
dropout_4/IdentityIdentitydense_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_4/Identity?
dense_7/MatMul/ReadVariableOpReadVariableOp,dense_7_matmul_readvariableop_dense_7_kernel*
_output_shapes
:	?@*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldropout_4/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp+dense_7_biasadd_readvariableop_dense_7_bias*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_7/Relu?
dropout_5/IdentityIdentitydense_7/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout_5/Identity?
dense_8/MatMul/ReadVariableOpReadVariableOp,dense_8_matmul_readvariableop_dense_8_kernel*
_output_shapes

:@*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldropout_5/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp+dense_8_biasadd_readvariableop_dense_8_bias*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAddy
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_8/Sigmoidn
IdentityIdentitydense_8/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_14_layer_call_and_return_conditional_losses_7717

inputs@
&conv2d_readvariableop_conv2d_14_kernel:@@3
%biasadd_readvariableop_conv2d_14_bias:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_14_kernel*&
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
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_14_bias*
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
?
a
(__inference_dropout_4_layer_call_fn_7864

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
GPU2*0J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_58792
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
?
?
C__inference_conv2d_11_layer_call_and_return_conditional_losses_5533

inputs@
&conv2d_readvariableop_conv2d_11_kernel: 3
%biasadd_readvariableop_conv2d_11_bias: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_11_kernel*&
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
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_11_bias*
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
C__inference_conv2d_10_layer_call_and_return_conditional_losses_7213

inputs@
&conv2d_readvariableop_conv2d_10_kernel:3
%biasadd_readvariableop_conv2d_10_bias:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_10_kernel*&
_output_shapes
:*
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
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_10_bias*
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
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_13_layer_call_and_return_conditional_losses_7591

inputs@
&conv2d_readvariableop_conv2d_13_kernel:0@3
%biasadd_readvariableop_conv2d_13_bias:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_13_kernel*&
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
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_13_bias*
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
?
?
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7681

inputs9
+readvariableop_batch_normalization_13_gamma:@:
,readvariableop_1_batch_normalization_13_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_13_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_13_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance*
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
?
?
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7807

inputs9
+readvariableop_batch_normalization_14_gamma:@:
,readvariableop_1_batch_normalization_14_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_14_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_14_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance*
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
?

?
5__inference_batch_normalization_10_layer_call_fn_7240

inputs*
batch_normalization_10_gamma:)
batch_normalization_10_beta:0
"batch_normalization_10_moving_mean:4
&batch_normalization_10_moving_variance:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_55162
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
?
?
C__inference_conv2d_14_layer_call_and_return_conditional_losses_5647

inputs@
&conv2d_readvariableop_conv2d_14_kernel:@@3
%biasadd_readvariableop_conv2d_14_bias:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_14_kernel*&
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
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_14_bias*
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
5__inference_batch_normalization_11_layer_call_fn_7366

inputs*
batch_normalization_11_gamma: )
batch_normalization_11_beta: 0
"batch_normalization_11_moving_mean: 4
&batch_normalization_11_moving_variance: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_55542
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
?
?
&__inference_dense_6_layer_call_fn_7843

inputs#
dense_6_kernel:???
dense_6_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_kerneldense_6_bias*
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
GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_56932
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
?
?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_5097

inputs9
+readvariableop_batch_normalization_12_gamma:0:
,readvariableop_1_batch_normalization_12_beta:0P
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean:0V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?
a
C__inference_dropout_5_layer_call_and_return_conditional_losses_7914

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
?
?
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_5630

inputs9
+readvariableop_batch_normalization_13_gamma:@:
,readvariableop_1_batch_normalization_13_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_13_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_13_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance*
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
?
A__inference_dense_8_layer_call_and_return_conditional_losses_5737

inputs6
$matmul_readvariableop_dense_8_kernel:@1
#biasadd_readvariableop_dense_8_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_8_kernel*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_8_bias*
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
?
?
(__inference_conv2d_14_layer_call_fn_7706

inputs*
conv2d_14_kernel:@@
conv2d_14_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_14_kernelconv2d_14_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_56472
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
?i
?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_6788
input_24
conv2d_10_conv2d_10_kernel:&
conv2d_10_conv2d_10_bias:A
3batch_normalization_10_batch_normalization_10_gamma:@
2batch_normalization_10_batch_normalization_10_beta:G
9batch_normalization_10_batch_normalization_10_moving_mean:K
=batch_normalization_10_batch_normalization_10_moving_variance:4
conv2d_11_conv2d_11_kernel: &
conv2d_11_conv2d_11_bias: A
3batch_normalization_11_batch_normalization_11_gamma: @
2batch_normalization_11_batch_normalization_11_beta: G
9batch_normalization_11_batch_normalization_11_moving_mean: K
=batch_normalization_11_batch_normalization_11_moving_variance: 4
conv2d_12_conv2d_12_kernel: 0&
conv2d_12_conv2d_12_bias:0A
3batch_normalization_12_batch_normalization_12_gamma:0@
2batch_normalization_12_batch_normalization_12_beta:0G
9batch_normalization_12_batch_normalization_12_moving_mean:0K
=batch_normalization_12_batch_normalization_12_moving_variance:04
conv2d_13_conv2d_13_kernel:0@&
conv2d_13_conv2d_13_bias:@A
3batch_normalization_13_batch_normalization_13_gamma:@@
2batch_normalization_13_batch_normalization_13_beta:@G
9batch_normalization_13_batch_normalization_13_moving_mean:@K
=batch_normalization_13_batch_normalization_13_moving_variance:@4
conv2d_14_conv2d_14_kernel:@@&
conv2d_14_conv2d_14_bias:@A
3batch_normalization_14_batch_normalization_14_gamma:@@
2batch_normalization_14_batch_normalization_14_beta:@G
9batch_normalization_14_batch_normalization_14_moving_mean:@K
=batch_normalization_14_batch_normalization_14_moving_variance:@+
dense_6_dense_6_kernel:???#
dense_6_dense_6_bias:	?)
dense_7_dense_7_kernel:	?@"
dense_7_dense_7_bias:@(
dense_8_dense_8_kernel:@"
dense_8_dense_8_bias:
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_10_conv2d_10_kernelconv2d_10_conv2d_10_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_54952#
!conv2d_10/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:03batch_normalization_10_batch_normalization_10_gamma2batch_normalization_10_batch_normalization_10_beta9batch_normalization_10_batch_normalization_10_moving_mean=batch_normalization_10_batch_normalization_10_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_632320
.batch_normalization_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_55332#
!conv2d_11/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:03batch_normalization_11_batch_normalization_11_gamma2batch_normalization_11_batch_normalization_11_beta9batch_normalization_11_batch_normalization_11_moving_mean=batch_normalization_11_batch_normalization_11_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_623220
.batch_normalization_11/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_55712#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:03batch_normalization_12_batch_normalization_12_gamma2batch_normalization_12_batch_normalization_12_beta9batch_normalization_12_batch_normalization_12_moving_mean=batch_normalization_12_batch_normalization_12_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_614120
.batch_normalization_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_56092#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:03batch_normalization_13_batch_normalization_13_gamma2batch_normalization_13_batch_normalization_13_beta9batch_normalization_13_batch_normalization_13_moving_mean=batch_normalization_13_batch_normalization_13_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_605020
.batch_normalization_13/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_14_conv2d_14_kernelconv2d_14_conv2d_14_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_56472#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:03batch_normalization_14_batch_normalization_14_gamma2batch_normalization_14_batch_normalization_14_beta9batch_normalization_14_batch_normalization_14_moving_mean=batch_normalization_14_batch_normalization_14_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_595920
.batch_normalization_14/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_56802
flatten_2/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_dense_6_kerneldense_6_dense_6_bias*
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
GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_56932!
dense_6/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_58792#
!dropout_4/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_7_dense_7_kerneldense_7_dense_7_bias*
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
A__inference_dense_7_layer_call_and_return_conditional_losses_57152!
dense_7/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
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
C__inference_dropout_5_layer_call_and_return_conditional_losses_58202#
!dropout_5/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_8_dense_8_kerneldense_8_dense_8_bias*
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
A__inference_dense_8_layer_call_and_return_conditional_losses_57372!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?

?
5__inference_batch_normalization_11_layer_call_fn_7357

inputs*
batch_normalization_11_gamma: )
batch_normalization_11_beta: 0
"batch_normalization_11_moving_mean: 4
&batch_normalization_11_moving_variance: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_49512
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
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7285

inputs9
+readvariableop_batch_normalization_10_gamma::
,readvariableop_1_batch_normalization_10_beta:P
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean:V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes
:*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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

?
5__inference_batch_normalization_10_layer_call_fn_7222

inputs*
batch_normalization_10_gamma:)
batch_normalization_10_beta:0
"batch_normalization_10_moving_mean:4
&batch_normalization_10_moving_variance:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_47692
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
C__inference_conv2d_10_layer_call_and_return_conditional_losses_5495

inputs@
&conv2d_readvariableop_conv2d_10_kernel:3
%biasadd_readvariableop_conv2d_10_bias:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_10_kernel*&
_output_shapes
:*
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
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_10_bias*
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
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?j
?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_6484

inputs4
conv2d_10_conv2d_10_kernel:&
conv2d_10_conv2d_10_bias:A
3batch_normalization_10_batch_normalization_10_gamma:@
2batch_normalization_10_batch_normalization_10_beta:G
9batch_normalization_10_batch_normalization_10_moving_mean:K
=batch_normalization_10_batch_normalization_10_moving_variance:4
conv2d_11_conv2d_11_kernel: &
conv2d_11_conv2d_11_bias: A
3batch_normalization_11_batch_normalization_11_gamma: @
2batch_normalization_11_batch_normalization_11_beta: G
9batch_normalization_11_batch_normalization_11_moving_mean: K
=batch_normalization_11_batch_normalization_11_moving_variance: 4
conv2d_12_conv2d_12_kernel: 0&
conv2d_12_conv2d_12_bias:0A
3batch_normalization_12_batch_normalization_12_gamma:0@
2batch_normalization_12_batch_normalization_12_beta:0G
9batch_normalization_12_batch_normalization_12_moving_mean:0K
=batch_normalization_12_batch_normalization_12_moving_variance:04
conv2d_13_conv2d_13_kernel:0@&
conv2d_13_conv2d_13_bias:@A
3batch_normalization_13_batch_normalization_13_gamma:@@
2batch_normalization_13_batch_normalization_13_beta:@G
9batch_normalization_13_batch_normalization_13_moving_mean:@K
=batch_normalization_13_batch_normalization_13_moving_variance:@4
conv2d_14_conv2d_14_kernel:@@&
conv2d_14_conv2d_14_bias:@A
3batch_normalization_14_batch_normalization_14_gamma:@@
2batch_normalization_14_batch_normalization_14_beta:@G
9batch_normalization_14_batch_normalization_14_moving_mean:@K
=batch_normalization_14_batch_normalization_14_moving_variance:@+
dense_6_dense_6_kernel:???#
dense_6_dense_6_bias:	?)
dense_7_dense_7_kernel:	?@"
dense_7_dense_7_bias:@(
dense_8_dense_8_kernel:@"
dense_8_dense_8_bias:
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_conv2d_10_kernelconv2d_10_conv2d_10_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_54952#
!conv2d_10/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:03batch_normalization_10_batch_normalization_10_gamma2batch_normalization_10_batch_normalization_10_beta9batch_normalization_10_batch_normalization_10_moving_mean=batch_normalization_10_batch_normalization_10_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_632320
.batch_normalization_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_55332#
!conv2d_11/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:03batch_normalization_11_batch_normalization_11_gamma2batch_normalization_11_batch_normalization_11_beta9batch_normalization_11_batch_normalization_11_moving_mean=batch_normalization_11_batch_normalization_11_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_623220
.batch_normalization_11/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_55712#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:03batch_normalization_12_batch_normalization_12_gamma2batch_normalization_12_batch_normalization_12_beta9batch_normalization_12_batch_normalization_12_moving_mean=batch_normalization_12_batch_normalization_12_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_614120
.batch_normalization_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_56092#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:03batch_normalization_13_batch_normalization_13_gamma2batch_normalization_13_batch_normalization_13_beta9batch_normalization_13_batch_normalization_13_moving_mean=batch_normalization_13_batch_normalization_13_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_605020
.batch_normalization_13/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_14_conv2d_14_kernelconv2d_14_conv2d_14_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_56472#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:03batch_normalization_14_batch_normalization_14_gamma2batch_normalization_14_batch_normalization_14_beta9batch_normalization_14_batch_normalization_14_moving_mean=batch_normalization_14_batch_normalization_14_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_595920
.batch_normalization_14/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_56802
flatten_2/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_dense_6_kerneldense_6_dense_6_bias*
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
GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_56932!
dense_6/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_58792#
!dropout_4/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_7_dense_7_kerneldense_7_dense_7_bias*
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
A__inference_dense_7_layer_call_and_return_conditional_losses_57152!
dense_7/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
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
C__inference_dropout_5_layer_call_and_return_conditional_losses_58202#
!dropout_5/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_8_dense_8_kerneldense_8_dense_8_bias*
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
A__inference_dense_8_layer_call_and_return_conditional_losses_57372!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_5389

inputs9
+readvariableop_batch_normalization_14_gamma:@:
,readvariableop_1_batch_normalization_14_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_14_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_14_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?
?
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7825

inputs9
+readvariableop_batch_normalization_14_gamma:@:
,readvariableop_1_batch_normalization_14_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_14_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_14_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_14_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_14_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?
?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_4951

inputs9
+readvariableop_batch_normalization_11_gamma: :
,readvariableop_1_batch_normalization_11_beta: P
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean: V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
5__inference_batch_normalization_14_layer_call_fn_7753

inputs*
batch_normalization_14_gamma:@)
batch_normalization_14_beta:@0
"batch_normalization_14_moving_mean:@4
&batch_normalization_14_moving_variance:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_14_gammabatch_normalization_14_beta"batch_normalization_14_moving_mean&batch_normalization_14_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_59592
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
?g
?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_5742

inputs4
conv2d_10_conv2d_10_kernel:&
conv2d_10_conv2d_10_bias:A
3batch_normalization_10_batch_normalization_10_gamma:@
2batch_normalization_10_batch_normalization_10_beta:G
9batch_normalization_10_batch_normalization_10_moving_mean:K
=batch_normalization_10_batch_normalization_10_moving_variance:4
conv2d_11_conv2d_11_kernel: &
conv2d_11_conv2d_11_bias: A
3batch_normalization_11_batch_normalization_11_gamma: @
2batch_normalization_11_batch_normalization_11_beta: G
9batch_normalization_11_batch_normalization_11_moving_mean: K
=batch_normalization_11_batch_normalization_11_moving_variance: 4
conv2d_12_conv2d_12_kernel: 0&
conv2d_12_conv2d_12_bias:0A
3batch_normalization_12_batch_normalization_12_gamma:0@
2batch_normalization_12_batch_normalization_12_beta:0G
9batch_normalization_12_batch_normalization_12_moving_mean:0K
=batch_normalization_12_batch_normalization_12_moving_variance:04
conv2d_13_conv2d_13_kernel:0@&
conv2d_13_conv2d_13_bias:@A
3batch_normalization_13_batch_normalization_13_gamma:@@
2batch_normalization_13_batch_normalization_13_beta:@G
9batch_normalization_13_batch_normalization_13_moving_mean:@K
=batch_normalization_13_batch_normalization_13_moving_variance:@4
conv2d_14_conv2d_14_kernel:@@&
conv2d_14_conv2d_14_bias:@A
3batch_normalization_14_batch_normalization_14_gamma:@@
2batch_normalization_14_batch_normalization_14_beta:@G
9batch_normalization_14_batch_normalization_14_moving_mean:@K
=batch_normalization_14_batch_normalization_14_moving_variance:@+
dense_6_dense_6_kernel:???#
dense_6_dense_6_bias:	?)
dense_7_dense_7_kernel:	?@"
dense_7_dense_7_bias:@(
dense_8_dense_8_kernel:@"
dense_8_dense_8_bias:
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_conv2d_10_kernelconv2d_10_conv2d_10_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_54952#
!conv2d_10/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:03batch_normalization_10_batch_normalization_10_gamma2batch_normalization_10_batch_normalization_10_beta9batch_normalization_10_batch_normalization_10_moving_mean=batch_normalization_10_batch_normalization_10_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_551620
.batch_normalization_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_55332#
!conv2d_11/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:03batch_normalization_11_batch_normalization_11_gamma2batch_normalization_11_batch_normalization_11_beta9batch_normalization_11_batch_normalization_11_moving_mean=batch_normalization_11_batch_normalization_11_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_555420
.batch_normalization_11/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_55712#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:03batch_normalization_12_batch_normalization_12_gamma2batch_normalization_12_batch_normalization_12_beta9batch_normalization_12_batch_normalization_12_moving_mean=batch_normalization_12_batch_normalization_12_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_559220
.batch_normalization_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_56092#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:03batch_normalization_13_batch_normalization_13_gamma2batch_normalization_13_batch_normalization_13_beta9batch_normalization_13_batch_normalization_13_moving_mean=batch_normalization_13_batch_normalization_13_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_563020
.batch_normalization_13/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_14_conv2d_14_kernelconv2d_14_conv2d_14_bias*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_56472#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:03batch_normalization_14_batch_normalization_14_gamma2batch_normalization_14_batch_normalization_14_beta9batch_normalization_14_batch_normalization_14_moving_mean=batch_normalization_14_batch_normalization_14_moving_variance*
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
GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_566820
.batch_normalization_14/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_56802
flatten_2/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_dense_6_kerneldense_6_dense_6_bias*
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
GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_56932!
dense_6/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_57022
dropout_4/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_7_dense_7_kerneldense_7_dense_7_bias*
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
A__inference_dense_7_layer_call_and_return_conditional_losses_57152!
dense_7/StatefulPartitionedCall?
dropout_5/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
C__inference_dropout_5_layer_call_and_return_conditional_losses_57242
dropout_5/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_8_dense_8_kerneldense_8_dense_8_bias*
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
A__inference_dense_8_layer_call_and_return_conditional_losses_57372!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7447

inputs9
+readvariableop_batch_normalization_11_gamma: :
,readvariableop_1_batch_normalization_11_beta: P
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean: V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?
b
C__inference_dropout_4_layer_call_and_return_conditional_losses_7881

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
?
?
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7699

inputs9
+readvariableop_batch_normalization_13_gamma:@:
,readvariableop_1_batch_normalization_13_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_13_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_13_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?
?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5554

inputs9
+readvariableop_batch_normalization_11_gamma: :
,readvariableop_1_batch_normalization_11_beta: P
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean: V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
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
?
?
&__inference_dense_8_layer_call_fn_7933

inputs 
dense_8_kernel:@
dense_8_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_kerneldense_8_bias*
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
A__inference_dense_8_layer_call_and_return_conditional_losses_57372
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
?
?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_5061

inputs9
+readvariableop_batch_normalization_12_gamma:0:
,readvariableop_1_batch_normalization_12_beta:0P
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean:0V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:0*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
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
?
?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_4915

inputs9
+readvariableop_batch_normalization_11_gamma: :
,readvariableop_1_batch_normalization_11_beta: P
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean: V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes
: *
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
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
?

b
C__inference_dropout_5_layer_call_and_return_conditional_losses_7926

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
?
?
"__inference_signature_wrapper_6831
input_2*
conv2d_10_kernel:
conv2d_10_bias:*
batch_normalization_10_gamma:)
batch_normalization_10_beta:0
"batch_normalization_10_moving_mean:4
&batch_normalization_10_moving_variance:*
conv2d_11_kernel: 
conv2d_11_bias: *
batch_normalization_11_gamma: )
batch_normalization_11_beta: 0
"batch_normalization_11_moving_mean: 4
&batch_normalization_11_moving_variance: *
conv2d_12_kernel: 0
conv2d_12_bias:0*
batch_normalization_12_gamma:0)
batch_normalization_12_beta:00
"batch_normalization_12_moving_mean:04
&batch_normalization_12_moving_variance:0*
conv2d_13_kernel:0@
conv2d_13_bias:@*
batch_normalization_13_gamma:@)
batch_normalization_13_beta:@0
"batch_normalization_13_moving_mean:@4
&batch_normalization_13_moving_variance:@*
conv2d_14_kernel:@@
conv2d_14_bias:@*
batch_normalization_14_gamma:@)
batch_normalization_14_beta:@0
"batch_normalization_14_moving_mean:@4
&batch_normalization_14_moving_variance:@#
dense_6_kernel:???
dense_6_bias:	?!
dense_7_kernel:	?@
dense_7_bias:@ 
dense_8_kernel:@
dense_8_bias:
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_10_kernelconv2d_10_biasbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_varianceconv2d_11_kernelconv2d_11_biasbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_varianceconv2d_12_kernelconv2d_12_biasbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_varianceconv2d_13_kernelconv2d_13_biasbatch_normalization_13_gammabatch_normalization_13_beta"batch_normalization_13_moving_mean&batch_normalization_13_moving_varianceconv2d_14_kernelconv2d_14_biasbatch_normalization_14_gammabatch_normalization_14_beta"batch_normalization_14_moving_mean&batch_normalization_14_moving_variancedense_6_kerneldense_6_biasdense_7_kerneldense_7_biasdense_8_kerneldense_8_bias*0
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
__inference__wrapped_model_47472
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
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
b
C__inference_dropout_5_layer_call_and_return_conditional_losses_5820

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
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7663

inputs9
+readvariableop_batch_normalization_13_gamma:@:
,readvariableop_1_batch_normalization_13_beta:@P
Bfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean:@V
Hfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1?
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_13_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp?
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_13_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance*
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
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_13_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_13_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
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
?
?
8__inference_single_image_simple_model_layer_call_fn_6913

inputs*
conv2d_10_kernel:
conv2d_10_bias:*
batch_normalization_10_gamma:)
batch_normalization_10_beta:0
"batch_normalization_10_moving_mean:4
&batch_normalization_10_moving_variance:*
conv2d_11_kernel: 
conv2d_11_bias: *
batch_normalization_11_gamma: )
batch_normalization_11_beta: 0
"batch_normalization_11_moving_mean: 4
&batch_normalization_11_moving_variance: *
conv2d_12_kernel: 0
conv2d_12_bias:0*
batch_normalization_12_gamma:0)
batch_normalization_12_beta:00
"batch_normalization_12_moving_mean:04
&batch_normalization_12_moving_variance:0*
conv2d_13_kernel:0@
conv2d_13_bias:@*
batch_normalization_13_gamma:@)
batch_normalization_13_beta:@0
"batch_normalization_13_moving_mean:@4
&batch_normalization_13_moving_variance:@*
conv2d_14_kernel:@@
conv2d_14_bias:@*
batch_normalization_14_gamma:@)
batch_normalization_14_beta:@0
"batch_normalization_14_moving_mean:@4
&batch_normalization_14_moving_variance:@#
dense_6_kernel:???
dense_6_bias:	?!
dense_7_kernel:	?@
dense_7_bias:@ 
dense_8_kernel:@
dense_8_bias:
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_kernelconv2d_10_biasbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_varianceconv2d_11_kernelconv2d_11_biasbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_varianceconv2d_12_kernelconv2d_12_biasbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_varianceconv2d_13_kernelconv2d_13_biasbatch_normalization_13_gammabatch_normalization_13_beta"batch_normalization_13_moving_mean&batch_normalization_13_moving_varianceconv2d_14_kernelconv2d_14_biasbatch_normalization_14_gammabatch_normalization_14_beta"batch_normalization_14_moving_mean&batch_normalization_14_moving_variancedense_6_kerneldense_6_biasdense_7_kerneldense_7_biasdense_8_kerneldense_8_bias*0
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
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_64842
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
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
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
input_2:
serving_default_input_2:0???????????;
dense_80
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
axis
	gamma
 beta
!moving_mean
"moving_variance
#	variables
$trainable_variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
<axis
	=gamma
>beta
?moving_mean
@moving_variance
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?m? m?'m?(m?.m?/m?6m?7m?=m?>m?Em?Fm?Lm?Mm?Tm?Um?[m?\m?gm?hm?qm?rm?{m?|m?v?v?v? v?'v?(v?.v?/v?6v?7v?=v?>v?Ev?Fv?Lv?Mv?Tv?Uv?[v?\v?gv?hv?qv?rv?{v?|v?"
	optimizer
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
 "
trackable_list_wrapper
?
?metrics
	variables
trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(2conv2d_10/kernel
:2conv2d_10/bias
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
	variables
trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_10/gamma
):'2batch_normalization_10/beta
2:0 (2"batch_normalization_10/moving_mean
6:4 (2&batch_normalization_10/moving_variance
<
0
 1
!2
"3"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
#	variables
$trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
%regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_11/kernel
: 2conv2d_11/bias
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
)	variables
*trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
+regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_11/gamma
):' 2batch_normalization_11/beta
2:0  (2"batch_normalization_11/moving_mean
6:4  (2&batch_normalization_11/moving_variance
<
.0
/1
02
13"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
2	variables
3trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
4regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 02conv2d_12/kernel
:02conv2d_12/bias
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
8	variables
9trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
:regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(02batch_normalization_12/gamma
):'02batch_normalization_12/beta
2:00 (2"batch_normalization_12/moving_mean
6:40 (2&batch_normalization_12/moving_variance
<
=0
>1
?2
@3"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
A	variables
Btrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
Cregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(0@2conv2d_13/kernel
:@2conv2d_13/bias
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
G	variables
Htrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
Iregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_13/gamma
):'@2batch_normalization_13/beta
2:0@ (2"batch_normalization_13/moving_mean
6:4@ (2&batch_normalization_13/moving_variance
<
L0
M1
N2
O3"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
P	variables
Qtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
Rregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_14/kernel
:@2conv2d_14/bias
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
V	variables
Wtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
Xregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_14/gamma
):'@2batch_normalization_14/beta
2:0@ (2"batch_normalization_14/moving_mean
6:4@ (2&batch_normalization_14/moving_variance
<
[0
\1
]2
^3"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
_	variables
`trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
aregularization_losses
?layers
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
c	variables
dtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
eregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!???2dense_6/kernel
:?2dense_6/bias
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
i	variables
jtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
kregularization_losses
?layers
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
m	variables
ntrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
oregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?@2dense_7/kernel
:@2dense_7/bias
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
s	variables
ttrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
uregularization_losses
?layers
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
w	variables
xtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
yregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_8/kernel
:2dense_8/bias
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
}	variables
~trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2training_2/Adam/iter
 : (2training_2/Adam/beta_1
 : (2training_2/Adam/beta_2
: (2training_2/Adam/decay
':% (2training_2/Adam/learning_rate
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
.
!0
"1"
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
.
00
11"
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
.
?0
@1"
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
.
N0
O1"
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
.
]0
^1"
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
:  (2total_1
:  (2count_1
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
::82"training_2/Adam/conv2d_10/kernel/m
,:*2 training_2/Adam/conv2d_10/bias/m
::82.training_2/Adam/batch_normalization_10/gamma/m
9:72-training_2/Adam/batch_normalization_10/beta/m
::8 2"training_2/Adam/conv2d_11/kernel/m
,:* 2 training_2/Adam/conv2d_11/bias/m
::8 2.training_2/Adam/batch_normalization_11/gamma/m
9:7 2-training_2/Adam/batch_normalization_11/beta/m
::8 02"training_2/Adam/conv2d_12/kernel/m
,:*02 training_2/Adam/conv2d_12/bias/m
::802.training_2/Adam/batch_normalization_12/gamma/m
9:702-training_2/Adam/batch_normalization_12/beta/m
::80@2"training_2/Adam/conv2d_13/kernel/m
,:*@2 training_2/Adam/conv2d_13/bias/m
::8@2.training_2/Adam/batch_normalization_13/gamma/m
9:7@2-training_2/Adam/batch_normalization_13/beta/m
::8@@2"training_2/Adam/conv2d_14/kernel/m
,:*@2 training_2/Adam/conv2d_14/bias/m
::8@2.training_2/Adam/batch_normalization_14/gamma/m
9:7@2-training_2/Adam/batch_normalization_14/beta/m
3:1???2 training_2/Adam/dense_6/kernel/m
+:)?2training_2/Adam/dense_6/bias/m
1:/	?@2 training_2/Adam/dense_7/kernel/m
*:(@2training_2/Adam/dense_7/bias/m
0:.@2 training_2/Adam/dense_8/kernel/m
*:(2training_2/Adam/dense_8/bias/m
::82"training_2/Adam/conv2d_10/kernel/v
,:*2 training_2/Adam/conv2d_10/bias/v
::82.training_2/Adam/batch_normalization_10/gamma/v
9:72-training_2/Adam/batch_normalization_10/beta/v
::8 2"training_2/Adam/conv2d_11/kernel/v
,:* 2 training_2/Adam/conv2d_11/bias/v
::8 2.training_2/Adam/batch_normalization_11/gamma/v
9:7 2-training_2/Adam/batch_normalization_11/beta/v
::8 02"training_2/Adam/conv2d_12/kernel/v
,:*02 training_2/Adam/conv2d_12/bias/v
::802.training_2/Adam/batch_normalization_12/gamma/v
9:702-training_2/Adam/batch_normalization_12/beta/v
::80@2"training_2/Adam/conv2d_13/kernel/v
,:*@2 training_2/Adam/conv2d_13/bias/v
::8@2.training_2/Adam/batch_normalization_13/gamma/v
9:7@2-training_2/Adam/batch_normalization_13/beta/v
::8@@2"training_2/Adam/conv2d_14/kernel/v
,:*@2 training_2/Adam/conv2d_14/bias/v
::8@2.training_2/Adam/batch_normalization_14/gamma/v
9:7@2-training_2/Adam/batch_normalization_14/beta/v
3:1???2 training_2/Adam/dense_6/kernel/v
+:)?2training_2/Adam/dense_6/bias/v
1:/	?@2 training_2/Adam/dense_7/kernel/v
*:(@2training_2/Adam/dense_7/bias/v
0:.@2 training_2/Adam/dense_8/kernel/v
*:(2training_2/Adam/dense_8/bias/v
?2?
8__inference_single_image_simple_model_layer_call_fn_5781
8__inference_single_image_simple_model_layer_call_fn_6872
8__inference_single_image_simple_model_layer_call_fn_6913
8__inference_single_image_simple_model_layer_call_fn_6676?
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
?2?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_7047
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_7195
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_6732
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_6788?
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
__inference__wrapped_model_4747input_2"?
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
?2?
(__inference_conv2d_10_layer_call_fn_7202?
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
C__inference_conv2d_10_layer_call_and_return_conditional_losses_7213?
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
5__inference_batch_normalization_10_layer_call_fn_7222
5__inference_batch_normalization_10_layer_call_fn_7231
5__inference_batch_normalization_10_layer_call_fn_7240
5__inference_batch_normalization_10_layer_call_fn_7249?
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
?2?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7267
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7285
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7303
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7321?
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
(__inference_conv2d_11_layer_call_fn_7328?
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
C__inference_conv2d_11_layer_call_and_return_conditional_losses_7339?
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
5__inference_batch_normalization_11_layer_call_fn_7348
5__inference_batch_normalization_11_layer_call_fn_7357
5__inference_batch_normalization_11_layer_call_fn_7366
5__inference_batch_normalization_11_layer_call_fn_7375?
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
?2?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7393
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7411
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7429
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7447?
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
(__inference_conv2d_12_layer_call_fn_7454?
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
C__inference_conv2d_12_layer_call_and_return_conditional_losses_7465?
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
5__inference_batch_normalization_12_layer_call_fn_7474
5__inference_batch_normalization_12_layer_call_fn_7483
5__inference_batch_normalization_12_layer_call_fn_7492
5__inference_batch_normalization_12_layer_call_fn_7501?
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
?2?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7519
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7537
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7555
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7573?
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
(__inference_conv2d_13_layer_call_fn_7580?
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
C__inference_conv2d_13_layer_call_and_return_conditional_losses_7591?
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
5__inference_batch_normalization_13_layer_call_fn_7600
5__inference_batch_normalization_13_layer_call_fn_7609
5__inference_batch_normalization_13_layer_call_fn_7618
5__inference_batch_normalization_13_layer_call_fn_7627?
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
?2?
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7645
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7663
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7681
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7699?
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
(__inference_conv2d_14_layer_call_fn_7706?
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
C__inference_conv2d_14_layer_call_and_return_conditional_losses_7717?
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
5__inference_batch_normalization_14_layer_call_fn_7726
5__inference_batch_normalization_14_layer_call_fn_7735
5__inference_batch_normalization_14_layer_call_fn_7744
5__inference_batch_normalization_14_layer_call_fn_7753?
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
?2?
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7771
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7789
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7807
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7825?
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
(__inference_flatten_2_layer_call_fn_7830?
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
C__inference_flatten_2_layer_call_and_return_conditional_losses_7836?
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
&__inference_dense_6_layer_call_fn_7843?
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
A__inference_dense_6_layer_call_and_return_conditional_losses_7854?
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
(__inference_dropout_4_layer_call_fn_7859
(__inference_dropout_4_layer_call_fn_7864?
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
C__inference_dropout_4_layer_call_and_return_conditional_losses_7869
C__inference_dropout_4_layer_call_and_return_conditional_losses_7881?
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
&__inference_dense_7_layer_call_fn_7888?
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
A__inference_dense_7_layer_call_and_return_conditional_losses_7899?
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
(__inference_dropout_5_layer_call_fn_7904
(__inference_dropout_5_layer_call_fn_7909?
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
C__inference_dropout_5_layer_call_and_return_conditional_losses_7914
C__inference_dropout_5_layer_call_and_return_conditional_losses_7926?
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
&__inference_dense_8_layer_call_fn_7933?
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
A__inference_dense_8_layer_call_and_return_conditional_losses_7944?
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
"__inference_signature_wrapper_6831input_2"?
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
__inference__wrapped_model_4747?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|:?7
0?-
+?(
input_2???????????
? "1?.
,
dense_8!?
dense_8??????????
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7267? !"M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7285? !"M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7303v !"=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7321v !"=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
5__inference_batch_normalization_10_layer_call_fn_7222? !"M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
5__inference_batch_normalization_10_layer_call_fn_7231? !"M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
5__inference_batch_normalization_10_layer_call_fn_7240i !"=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
5__inference_batch_normalization_10_layer_call_fn_7249i !"=?:
3?0
*?'
inputs???????????
p
? ""?????????????
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7393?./01M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7411?./01M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7429r./01;?8
1?.
(?%
inputs?????????HH 
p 
? "-?*
#? 
0?????????HH 
? ?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7447r./01;?8
1?.
(?%
inputs?????????HH 
p
? "-?*
#? 
0?????????HH 
? ?
5__inference_batch_normalization_11_layer_call_fn_7348?./01M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
5__inference_batch_normalization_11_layer_call_fn_7357?./01M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
5__inference_batch_normalization_11_layer_call_fn_7366e./01;?8
1?.
(?%
inputs?????????HH 
p 
? " ??????????HH ?
5__inference_batch_normalization_11_layer_call_fn_7375e./01;?8
1?.
(?%
inputs?????????HH 
p
? " ??????????HH ?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7519?=>?@M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7537?=>?@M?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7555r=>?@;?8
1?.
(?%
inputs?????????##0
p 
? "-?*
#? 
0?????????##0
? ?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7573r=>?@;?8
1?.
(?%
inputs?????????##0
p
? "-?*
#? 
0?????????##0
? ?
5__inference_batch_normalization_12_layer_call_fn_7474?=>?@M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
5__inference_batch_normalization_12_layer_call_fn_7483?=>?@M?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
5__inference_batch_normalization_12_layer_call_fn_7492e=>?@;?8
1?.
(?%
inputs?????????##0
p 
? " ??????????##0?
5__inference_batch_normalization_12_layer_call_fn_7501e=>?@;?8
1?.
(?%
inputs?????????##0
p
? " ??????????##0?
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7645?LMNOM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7663?LMNOM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7681rLMNO;?8
1?.
(?%
inputs?????????!!@
p 
? "-?*
#? 
0?????????!!@
? ?
P__inference_batch_normalization_13_layer_call_and_return_conditional_losses_7699rLMNO;?8
1?.
(?%
inputs?????????!!@
p
? "-?*
#? 
0?????????!!@
? ?
5__inference_batch_normalization_13_layer_call_fn_7600?LMNOM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
5__inference_batch_normalization_13_layer_call_fn_7609?LMNOM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
5__inference_batch_normalization_13_layer_call_fn_7618eLMNO;?8
1?.
(?%
inputs?????????!!@
p 
? " ??????????!!@?
5__inference_batch_normalization_13_layer_call_fn_7627eLMNO;?8
1?.
(?%
inputs?????????!!@
p
? " ??????????!!@?
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7771?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7789?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7807r[\]^;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
P__inference_batch_normalization_14_layer_call_and_return_conditional_losses_7825r[\]^;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
5__inference_batch_normalization_14_layer_call_fn_7726?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
5__inference_batch_normalization_14_layer_call_fn_7735?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
5__inference_batch_normalization_14_layer_call_fn_7744e[\]^;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
5__inference_batch_normalization_14_layer_call_fn_7753e[\]^;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
C__inference_conv2d_10_layer_call_and_return_conditional_losses_7213p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
(__inference_conv2d_10_layer_call_fn_7202c9?6
/?,
*?'
inputs???????????
? ""?????????????
C__inference_conv2d_11_layer_call_and_return_conditional_losses_7339n'(9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????HH 
? ?
(__inference_conv2d_11_layer_call_fn_7328a'(9?6
/?,
*?'
inputs???????????
? " ??????????HH ?
C__inference_conv2d_12_layer_call_and_return_conditional_losses_7465l677?4
-?*
(?%
inputs?????????HH 
? "-?*
#? 
0?????????##0
? ?
(__inference_conv2d_12_layer_call_fn_7454_677?4
-?*
(?%
inputs?????????HH 
? " ??????????##0?
C__inference_conv2d_13_layer_call_and_return_conditional_losses_7591lEF7?4
-?*
(?%
inputs?????????##0
? "-?*
#? 
0?????????!!@
? ?
(__inference_conv2d_13_layer_call_fn_7580_EF7?4
-?*
(?%
inputs?????????##0
? " ??????????!!@?
C__inference_conv2d_14_layer_call_and_return_conditional_losses_7717lTU7?4
-?*
(?%
inputs?????????!!@
? "-?*
#? 
0?????????@
? ?
(__inference_conv2d_14_layer_call_fn_7706_TU7?4
-?*
(?%
inputs?????????!!@
? " ??????????@?
A__inference_dense_6_layer_call_and_return_conditional_losses_7854_gh1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? |
&__inference_dense_6_layer_call_fn_7843Rgh1?.
'?$
"?
inputs???????????
? "????????????
A__inference_dense_7_layer_call_and_return_conditional_losses_7899]qr0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? z
&__inference_dense_7_layer_call_fn_7888Pqr0?-
&?#
!?
inputs??????????
? "??????????@?
A__inference_dense_8_layer_call_and_return_conditional_losses_7944\{|/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? y
&__inference_dense_8_layer_call_fn_7933O{|/?,
%?"
 ?
inputs?????????@
? "???????????
C__inference_dropout_4_layer_call_and_return_conditional_losses_7869^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
C__inference_dropout_4_layer_call_and_return_conditional_losses_7881^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? }
(__inference_dropout_4_layer_call_fn_7859Q4?1
*?'
!?
inputs??????????
p 
? "???????????}
(__inference_dropout_4_layer_call_fn_7864Q4?1
*?'
!?
inputs??????????
p
? "????????????
C__inference_dropout_5_layer_call_and_return_conditional_losses_7914\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
C__inference_dropout_5_layer_call_and_return_conditional_losses_7926\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? {
(__inference_dropout_5_layer_call_fn_7904O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@{
(__inference_dropout_5_layer_call_fn_7909O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
C__inference_flatten_2_layer_call_and_return_conditional_losses_7836b7?4
-?*
(?%
inputs?????????@
? "'?$
?
0???????????
? ?
(__inference_flatten_2_layer_call_fn_7830U7?4
-?*
(?%
inputs?????????@
? "?????????????
"__inference_signature_wrapper_6831?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|E?B
? 
;?8
6
input_2+?(
input_2???????????"1?.
,
dense_8!?
dense_8??????????
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_6732?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|B??
8?5
+?(
input_2???????????
p 

 
? "%?"
?
0?????????
? ?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_6788?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|B??
8?5
+?(
input_2???????????
p

 
? "%?"
?
0?????????
? ?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_7047?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
S__inference_single_image_simple_model_layer_call_and_return_conditional_losses_7195?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
8__inference_single_image_simple_model_layer_call_fn_5781?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|B??
8?5
+?(
input_2???????????
p 

 
? "???????????
8__inference_single_image_simple_model_layer_call_fn_6676?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|B??
8?5
+?(
input_2???????????
p

 
? "???????????
8__inference_single_image_simple_model_layer_call_fn_6872?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
8__inference_single_image_simple_model_layer_call_fn_6913?$ !"'(./0167=>?@EFLMNOTU[\]^ghqr{|A?>
7?4
*?'
inputs???????????
p

 
? "??????????