import tensorflow_datasets as tfds
import tensorflow as tf

class Data :
    def __init__(self,dataset,batch_size):
        dataset, metadata = tfds.load(dataset, as_supervised=True, with_info=True)
        self.__train_dataset, self.__test_dataset = dataset['train'], dataset['test']

        self.__set_class_names(metadata.features['label'].names)
        # class_names = metadata.features['label'].names
        print("Class names: {}".format(self.__class_names))

        self.__num_train_examples = metadata.splits['train'].num_examples
        self.__num_test_examples = metadata.splits['test'].num_examples
        print("Number of training examples: {}".format(self.__num_train_examples))
        print("Number of test examples:     {}".format(self.__num_test_examples))

        def normalize(images, labels):
            images = tf.cast(images, tf.float32)
            images /= 255
            return images, labels
        
        self.__train_dataset =  self.__train_dataset.map(normalize)
        self.__test_dataset  =  self.__test_dataset.map(normalize)

        self.__train_dataset =  self.__train_dataset.cache()
        self.__test_dataset  =  self.__test_dataset.cache()

        self.__train_dataset = self.__train_dataset.cache().repeat().shuffle(self.__num_train_examples).batch(batch_size)
        self.__test_dataset = self.__test_dataset.cache().batch(batch_size)

    def __get_train_dataset(self):
        return self.__train_dataset
      
    def __set_train_dataset(self,dataset):
        self.__train_dataset = dataset

    train_dataset = property(__get_train_dataset,__set_train_dataset)

    def __get_test_dataset(self):
        return self.__test_dataset
      
    def __set_test_dataset(self,dataset):
        self.__test_dataset = dataset

    test_dataset = property(__get_test_dataset,__set_test_dataset)

    def __get_class_names(self):
        return self.__class_names
    
    def __set_class_names(self,class_names):
        self.__class_names = class_names

    class_names = property(__get_class_names, __set_class_names)

    def __get_num_test_examples(self):
        return self.__num_test_examples
     
    def __set_num_test_examples(self,num_test_examples):
        self.__num_test_examples = num_test_examples

    num_test_examples = property(__get_num_test_examples, __set_num_test_examples)

    def __get_num_train_examples(self):
        return self.__num_train_examples
     
    def __set_num_train_examples(self,num_train_examples):
        self.__num_train_examples = num_train_examples

    num_train_examples = property(__get_num_train_examples, __set_num_train_examples)
