
hdfs:///user/hadoop/suanec/

/// ***************testSVM failed***************
/usr/lib/spark/bin/spark-submit \
        --master yarn \
        --class com.testSVM \
        --executor-memory 10G \
        --num-executors 1 \
        file:///home/hadoop/suanec/suae/workspace/TestSVM.jar \
        hdfs:///user/hadoop/suanec/sample_libsvm_data.txt

/// ***************Test_example_NN***************
/usr/lib/spark/bin/spark-submit \
        --master yarn \
        --class tests.Test_example_NN \
        --executor-memory 16G \
        --num-executors 6 \
        file:///home/hadoop/suanec/suae/workspace/NN.jar 

/// ***************testPackage***************
/usr/lib/spark/bin/spark-submit \
        --master yarn \
        --class Hello \
        --executor-memory 1G \
        --num-executors 1 \
        file:///home/hadoop/suanec/suae/workspace/TP.jar 

/// ***************MTrick***************
/usr/lib/spark/bin/spark-submit \
        --master yarn \
        --class tests.Test_MTrick \
        --executor-memory 16G \
        --num-executors 6 \
        file:///home/hadoop/suanec/suae/workspace/Temp.jar 

/// ***************  ***************
/usr/lib/spark/bin/spark-submit \
        --master yarn \
        --class org.deeplearning4j.examples.deepbelief.DeepAutoEncoderExample \
        --executor-memory 16G \
        --num-executors 6 \
        file:///home/hadoop/suanec/suae/workspace/deeplearning4j-examples-0.4-rc0-SNAPSHOT.jar 

/// ***************  ***************
/usr/lib/spark/bin/spark-submit \
        --master local[2] \
        --class org.gd.spark.opendl.example.spark.dATest \
        --executor-memory 16G \
        --num-executors 6 \
        file:///home/hadoop/suanec/suae/workspace/opendl-example-0.0.1-SNAPSHOT.jar 

/// ***************  ***************
/usr/lib/spark/bin/spark-submit \
        --master yarn \
        --class org.deeplearning4j.examples.deepbelief.DBNAutoEncoderExample \
        --executor-memory 16G \
        --num-executors 6 \
        file:///home/hadoop/suanec/suae/workspace/deeplearning4j-examples-0.4-rc0-SNAPSHOT.jar

/// *************** scene-classification-spark ***************
/usr/lib/spark/bin/spark-submit \
        --master yarn \
        --class org.deeplearning4j.App \
        --driver-memory 25G \
        --executor-memory 25G \
        file:///home/hadoop/suanec/suae/workspace/scene-classification-spark-1.0-SNAPSHOT.jar

/// *************** testDL4j-MnistClassification ***************
/usr/lib/spark/bin/spark-submit \
        --master yarn \
        --class MnistClassification \
        --jars file:///home/hadoop/suanec/suae/workspace/Dl4jAndNd4j.jar \
        --executor-memory 16G \
        --num-executors 6 \
        file:///home/hadoop/suanec/suae/workspace/testDL4j.jar

---------------------------------------------------------------------
scalac Sample
scala Sample
java -cp .;"scala-library.jar" Sample

javac Hello.java
jar -cvf Hello.jar Hello.class
jar -cvf classes.jar Foo.class Bar.class       ///  将两个class文件存档到一个名为classes.jar的存档文件中。
jar -cvfm classes.jar mymanifest -C foo/.      ///  用一个存在的清单文件'mymanifest'将foo/目录下所有文件存档到一个名为'classes.jar'的存档文件中
jar -umf MANIFEST.MF Hello.jar                 ///  更新了MANIFEST.MF 
Java -jar Hello.jar
jar cvfm test.jar manifest.mf test 
"C:\Program Files\scala\lib\scala-library.jar"
scalac -classpath .;spark-assem*.jar NN.scala NNModel.scala 
Main-Class: test(MANIFEST.MF)
---------------------------------------------------------------------

