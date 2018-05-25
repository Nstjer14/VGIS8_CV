from keras.models import load_model
import iris_face_merge_cnn_data_splitter as getData

train_iris_X =  getData.train_iris_X
train_face_X =  getData.train_face_X
train_label =  getData.train_label

test_iris_X =  getData.test_iris_X
test_face_X =  getData.test_face_X
test_label =  getData.test_label

validation_iris_X =  getData.validation_iris_X
validation_face_X =  getData.validation_face_X
validation_label =  getData.validation_label

model_good = load_model('saved_models\iris_cnn_test_20180522-161616_acc_99.77.h5')
mode_not_so_good = load_model('saved_models\\20180525-161700acc_74.14_iris_cnn_test.h5')

score = model_old.evaluate(test_iris_X, test_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)

'''
merged_model.add(Merge([iris_model, face_model], mode = 'concat'))
#merged_model.add(Dense(1024, activation='relu',))
#merged_model.add(Dropout(0.5))
#merged_model.add(Dense(1024, activation='relu'))
#merged_model.add(Dropout(0.5))
merged_model.add(Dense(num_classes, activation='sigmoid',))
'''