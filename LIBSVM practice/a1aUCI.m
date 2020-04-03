%Reads file into LIBSVM format
[label_vector, instance_matrix] = libsvmread('a1a.txt');

%Trains optimal paramaters- using linear kernel
options = 'b 1';
model = svmtrain(label_vector, instance_matrix, ''); %#ok<*SVMTRAIN>

%Reads test file into LIBSVM format
[label_vector_test, instance_matrix_test] = libsvmread('a1a.t');

%Checks accuracy on test set
[predict_vector, accuracy, ~] = svmpredict(label_vector_test, instance_matrix_test, model, '-q')
