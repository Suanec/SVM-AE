
//spark-submit --class test.test --executor-memory 10G --driver-memory 10G --master yarn --conf spark.kryoserializer.buffer.max.mb=200 --num-executors 6 /home/hadoop/fhtn/MAE.jar 4 20000
//spark-submit --class test.test --executor-memory 10G --driver-memory 10G --master yarn /home/hadoop/fhtn/SVM.jar 0.5 134
//val a = Array(798,970,963,979,958,982,964,987,993,991,997,989,984,987,985,997,909,940,774,627)
///home/hadoop/suanec/data/MTrick
//hadoop fs -rm /user/hadoop/suanec/MAE.jar
//hadoop fs -put MAE.jar /user/hadoop/suanec

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD 
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel


val data_path = "file:///home/hadoop/fhtn/MAEres/res2000.data"//"hdfs:///user/hadoop/suanec/QMDEncodeRes.data"
//val data_path = "file:///home/hadoop/fhtn/MAEres/res"+name.toString+".data"
val label_path = "file:///home/hadoop/fhtn/datafile.label"
//Logger.getRootLogger.setLevel(Level.ERROR)
/// Load train data
val data = sc.textFile(data_path)
val label = sc.textFile(label_path)
val datas = data.collect
val labels = label.collect
val vec = new Array[Array[Double]](labels.size)

/// Generate the train vec
val dataLength = datas.head.split(',').size
for( i <- 0 until vec.size){
  val tmp = new Array[Double](dataLength+1)
  val splits = datas(i).split(',')
  val dataDouble = new Array[Double](dataLength)
  tmp(0) = (labels(i).toDouble -1)
  for( j <- 0 until splits.size-1){
    dataDouble(j) = splits(j).toDouble
  }
  dataDouble.copyToArray(tmp,1)
  vec(i) = tmp
}

/// Generate the train RDD
val rd = sc.parallelize(vec,10)
val parsedData1 = rd.map{ line =>
  val features = line.tail
  /// Vector Dense
  // LabeledPoint(line(0).toInt, Vectors.dense(features))
  /// Vector Sparse
  //val Seq = features.zipWithIndex.map(e => (e._2,e._1)).filter(_._2 != 0).unzip
  LabeledPoint(line(0) ,Vectors.dense(features.toArray))

}

val randRate = 0.5//rate.toDouble/10
val randomSplits = parsedData1.randomSplit(Array(randRate,1-randRate),seed = 155l)
val parsedData = randomSplits(0)
val testData = randomSplits(1)

//val numIterations = 200 
//val model = SVMWithSGD.train(parsedData, numIterations) 
//val lamda = 0.01
//val model = NaiveBayes.train(parsedData, lamda) 
// val numClasses = 20
// val categoricalFeaturesInfo = Map[Int, Int]()
// val impurity = "gini"
// val maxDepth = 7
// val maxBins = 64

// val model = DecisionTree.trainClassifier(parsedData, numClasses, categoricalFeaturesInfo,
//   impurity, maxDepth, maxBins)
val labelAndPreds = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = labelAndPreds.filter( r => r._1 != r._2).count.toDouble / testData.count
println("Training Error = " + trainErr)
val trainPrecision = 1- trainErr
println("parsedData Precision = " + trainPrecision )
 
// accuracy(iter2)(iter-1) = trainPrecision
