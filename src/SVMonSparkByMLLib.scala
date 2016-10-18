-----------------------------------------------------------------------------

import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{
  Matrix => BM,
  CSCMatrix => BSM,
  DenseMatrix => BDM,
  Vector => BV,
  DenseVector => BDV,
  SparseVector => BSV,
  axpy => brzAxpy,
  svd => brzSvd,
  max => Bmax,
  min => Bmin,
  sum => Bsum
}
import scala.collection.mutable.ArrayBuffer
import java.io._
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

-----------------------------------------------------------------------------

/home/hadoop/suanec/data/MTrick
hadoop fs -rm /user/hadoop/suanec/QMDEncodeRes.data /user/hadoop/suanec/QMDEncodeTestRes.data
hadoop fs -put QMDEncodeRes.data QMDEncodeTestRes.data /user/hadoop/suanec

-----------------------------------------------------------------------------



-----------------------------------------------------------------------------

-----------------------------------------------------------------------------



-----------------------------------------------------------------------------

-----------------------------------------------------------------------------

/// MTrick SVM test five times
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD 
import org.apache.spark.mllib.util.MLUtils
val data_path = "file:///home/hadoop/suanec/data/MTrick/Train.data"
val label_path = "file:///home/hadoop/suanec/data/MTrick/Train.label"
val test_data_path = "file:///home/hadoop/suanec/data/MTrick/Test.data"
val test_label_path = "file:///home/hadoop/suanec/data/MTrick/Test.label"
/* hdfs:///user/hadoop/suanec
val data_path = "hdfs:///user/hadoop/suanec/Train.data"
val label_path = "hdfs:///user/hadoop/suanec/Train.label"
val test_data_path = "hdfs:///user/hadoop/suanec/Test.data"
val test_label_path = "hdfs:///user/hadoop/suanec/Test.label"
*/
val data = sc.textFile(data_path)
val label = sc.textFile(label_path)
val datas = data.collect
val labels = label.collect
// val arr = new ArrayBuffer[ArrayBuffer[String]]
val strArr = new Array[String] (datas(datas.size-1).split(',')(1).toInt)
val vecArr = new Array[Double] (7270)
val vecBuf = new ArrayBuffer[Array[Double]]
val dataBuf = new ArrayBuffer[Array[Double]]
var colNum = 0
var str = ""
var rowNum = 0
for(i <- 0 to datas.size-1){
  val splits = datas(i).split(',')
  if(rowNum != splits(1).toInt){
    strArr(splits(1).toInt-1) = splits(0) + ':' + splits(2)
    rowNum = splits(1).toInt
  }else{
    strArr(rowNum-1) += " " + splits(0) + ':' + splits(2)
  }
}/// 得到Index：Feature字符串数组strArr

for(i <- 0 to strArr.size-1){
  for(j <- 0 to vecArr.size-1) vecArr(j) = 0.0
  val splits = strArr(i).split(' ')
  for(j <- 0 to splits.size-1){
    val parts = splits(j).split(':')
    vecArr(parts(0).toInt-1) = parts(1).toDouble
  }
  val diff = vecArr.max - vecArr.min
  for(j <- 0 to vecArr.size-1){ 
    vecArr(j) = vecArr(j)/diff
  }
  vecBuf += vecArr
}/// 得到特征向量数组。
for(i <- 0 to vecBuf.size -1){
  // dataBuf += Array(labels(i).toDouble,vecBuf(i))
  dataBuf += new Array[Double]((vecBuf(i).size+1))
  dataBuf(i)(0) = labels(i).toDouble - 1.0
  print(dataBuf(i)(0) + "  ")
  vecBuf(i).copyToArray(dataBuf(i),1,vecBuf(i).size )
}/// 得到训练数据
val rd = sc.parallelize(dataBuf,10)
val parsedData1 = rd.map{ line =>
  val features = line.tail
  // for(i <- 1 to line.size-1) features(i-1) = line(i)
  /// Vector Dense
  // LabeledPoint(line(0).toInt, Vectors.dense(features))
  /// Vector Sparse
  val Seq = features.zipWithIndex.map(e => (e._2,e._1)).filter(_._2 != 0).unzip
  LabeledPoint(line(0).toInt,Vectors.sparse(features.size,Seq._1.toArray,Seq._2.toArray))
}

val randomSplits = parsedData1.randomSplit(Array(0.3,0.3,0.4),seed = 43L)
val parsedData = randomSplits(0)
val testData = randomSplits(1)
val numIterations = 20 
val model = SVMWithSGD.train(parsedData, numIterations)  

// Evaluate model on training examples and compute training error 
val labelAndPreds = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = labelAndPreds.filter( r => 
    r._1 != r._2).count.toDouble / parsedData.count
    println("Training Error = " + trainErr)
    val trainPrecision = 1- trainErr
    println("Precision = " + trainPrecision )

    /// 20次准确率74.1  0.5307402760351317
    /// 200次准确率75.28
    /// train分0.6和0.4 ， 89.91

-----------------------------------------------------------------------------

/usr/lib/spark/bin/spark-submit \
        --master yarn \
        --class test \
        --executor-memory 16G \
        --num-executors 6 \
        file:///home/hadoop/suanec/suae/workspace/1.jar


-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

/usr/lib/spark/bin/spark-submit \
        --master yarn \
        --class SVMMTrick \
        --executor-memory 16G \
        --num-executors 6 \
        file:///home/hadoop/suanec/suae/workspace/1.jar


-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

/usr/lib/spark/bin/spark-submit \
        --master yarn \
        --jars file:///home/hadoop/suanec/suae/workspace/SMLDLF.jar \
        --class TestAE \
        --executor-memory 28G \
        --num-executors 72 \
        file:///home/hadoop/suanec/suae/workspace/1.jar


-----------------------------------------------------------------------------

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD 
import org.apache.spark.mllib.util.MLUtils


val data_path = "file:///home/hadoop/suanec/data/MTrick/Train0to2k.data"
val label_path = "file:///home/hadoop/suanec/data/MTrick/Train.label"
val test_data_path = "file:///home/hadoop/suanec/data/MTrick/Test0to2k.data"
val test_label_path = "file:///home/hadoop/suanec/data/MTrick/Test.label"


val data = sc.textFile(data_path)
val labels = sc.textFile(label_path)
val data_arr = data.collect
val label_arr = labels.collect

val test_data = sc.textFile(test_data_path)
val test_data_arr = test_data.collect
val test_labels = sc.textFile(test_label_path)
val test_label_arr = test_labels.collect


val features = data_arr.map{ s =>
  val splits = s.split(' ')
  val indices = new ArrayBuffer[Int]
  val values = new ArrayBuffer[Double]
  splits.map{ x =>
    val splitsx = x.split(":")
    indices += splitsx.head.toInt
    values += splitsx.last.toDouble
  }
  val diff_values = values.max - values.min
  val res_values = values.map( i => (i - diff_values)/diff_values.toDouble )
  (indices.toArray,res_values.toArray)
}
val labeled_data = (0 until label_arr.size).map{ i => 
  (label_arr(i).toInt-1,features(i))
}
val point_data = labeled_data.map{ x => 
  LabeledPoint(x._1,Vectors.sparse(2000+1,x._2._1,x._2._2))
}

val test_features = test_data_arr.map{ s =>
  if(!s.isEmpty){
    val splits = s.split(' ')
    val indices = new ArrayBuffer[Int]
    val values = new ArrayBuffer[Double]
    splits.map{ x =>
      val splitsx = x.split(":")
      indices += splitsx.head.toInt
      values += splitsx.last.toDouble
    }
    val diff_values = values.max - values.min
    val res_values = values.map( i => (i - diff_values)/diff_values.toDouble )
    (indices.toArray,res_values.toArray)
  }else{
    val indices = new ArrayBuffer[Int]
    val values = new ArrayBuffer[Double]
    (indices.toArray, values.toArray)
  }
}

val test_labeled_data = (0 until test_label_arr.size).map{ i => 
  (test_label_arr(i).toInt-1,test_features(i))
}.filter(!_._2._1.isEmpty)
val test_point_data = test_labeled_data.map{ x => 
  LabeledPoint(x._1,Vectors.sparse(2000+1,x._2._1,x._2._2))
}

val parsedData = sc.parallelize(point_data)
val testData = sc.parallelize(test_point_data)
val numIterations = 20 
val model = SVMWithSGD.train(parsedData, numIterations)  

// Evaluate model on training examples and compute training error 
val labelAndPreds = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = labelAndPreds.filter( r => 
    r._1 != r._2).count.toDouble / parsedData.count
    println("Training Error = " + trainErr)
    val trainPrecision = 1- trainErr
    println("Precision = " + trainPrecision )  ///  59


-----------------------------------------------------------------------------


-----------------------------------------------------------------------------
val data = sc.textFile(test_data_path)
val label = sc.textFile(test_label_path)
val datas = data.collect
val labels = label.collect
// val arr = new ArrayBuffer[ArrayBuffer[String]]
val strArr = new Array[String] (datas(datas.size-1).split(',')(1).toInt)
val vecArr = new Array[Double] (7270)
val vecBuf = new ArrayBuffer[Array[Double]]
val dataBuf = new ArrayBuffer[Array[Double]]
var colNum = 0
var str = ""
var rowNum = 0
for(i <- 0 to datas.size-1){
  val splits = datas(i).split(',')
  if(rowNum != splits(1).toInt){
    strArr(splits(1).toInt-1) = splits(0) + ':' + splits(2)
    rowNum = splits(1).toInt
  }else{
    strArr(rowNum-1) += " " + splits(0) + ':' + splits(2)
  }
}/// 得到Index：Feature字符串数组strArr

for(i <- 0 to strArr.size-1){
  for(j <- 0 to vecArr.size-1) vecArr(j) = 0.0
  val splits = strArr(i).split(' ')
  for(j <- 0 to splits.size-1){
    val parts = splits(j).split(':')
    vecArr(parts(0).toInt-1) = parts(1).toDouble
  }
  val diff = vecArr.max - vecArr.min
  for(j <- 0 to vecArr.size-1){ 
    vecArr(j) = vecArr(j)/diff
  }
  vecBuf += vecArr
}/// 得到特征向量数组。
for(i <- 0 to vecBuf.size*2/3){
  // dataBuf += Array(labels(i).toDouble,vecBuf(i))
  dataBuf += new Array[Double]((vecBuf(i).size+1))
  dataBuf(i)(0) = labels(i).toDouble - 1.0
  print(dataBuf(i)(0) + "  ")
  vecBuf(i).copyToArray(dataBuf(i),1,vecBuf(i).size )
}/// 得到训练数据
val rd = sc.parallelize(dataBuf,10)
val parsedData1 = rd.map{ line =>
  val features = line.tail
  // for(i <- 1 to line.size-1) features(i-1) = line(i)
  /// Vector Dense
  // LabeledPoint(line(0).toInt, Vectors.dense(features))
  /// Vector Sparse
  val Seq = features.zipWithIndex.map(e => (e._2,e._1)).filter(_._2 != 0).unzip
  LabeledPoint(line(0).toInt,Vectors.sparse(features.size,Seq._1.toArray,Seq._2.toArray))
}
val parsedData = parsedData1.randomSplit(Array(0.1,0.9),seed = 11L)(0)
// Evaluate model on training examples and compute training error 
val labelAndPreds = parsedData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = labelAndPreds.filter( r => 
    r._1 != r._2).count.toDouble / parsedData.count
    println("Training Error = " + trainErr)
    println("Precision = " + (1-trainErr) )

-----------------------------------------------------------------------------

-----------------------------------------------------------------------------
val data = sc.textFile(test_data_path)
val label = sc.textFile(test_label_path)
val datas = data.collect
val labels = label.collect
// val arr = new ArrayBuffer[ArrayBuffer[String]]
val strArr = new Array[String] (datas(datas.size-1).split(',')(1).toInt)
val vecArr = new Array[Double] (7270)
val vecBuf = new ArrayBuffer[Array[Double]]
val dataBuf = new ArrayBuffer[Array[Double]]
var colNum = 0
var str = ""
var rowNum = 0
for(i <- 0 to datas.size-1){
  val splits = datas(i).split(',')
  if(rowNum != splits(1).toInt){
    strArr(splits(1).toInt-1) = splits(0) + ':' + splits(2)
    rowNum = splits(1).toInt
  }else{
    strArr(rowNum-1) += " " + splits(0) + ':' + splits(2)
  }
}/// 得到Index：Feature字符串数组strArr

for(i <- 0 to strArr.size-1){
  for(j <- 0 to vecArr.size-1) vecArr(j) = 0.0
  val splits = strArr(i).split(' ')
  for(j <- 0 to splits.size-1){
    val parts = splits(j).split(':')
    vecArr(parts(0).toInt-1) = parts(1).toDouble
  }
  val diff = vecArr.max - vecArr.min
  for(j <- 0 to vecArr.size-1){ 
    vecArr(j) = vecArr(j)/diff
  }
  vecBuf += vecArr
}/// 得到特征向量数组。
for(i <- 0 to vecBuf.size*2/3){
  // dataBuf += Array(labels(i).toDouble,vecBuf(i))
  dataBuf += new Array[Double]((vecBuf(i).size+1))
  dataBuf(i)(0) = labels(i).toDouble - 1.0
  print(dataBuf(i)(0) + "  ")
  vecBuf(i).copyToArray(dataBuf(i),1,vecBuf(i).size )
}/// 得到训练数据
val rd = sc.parallelize(dataBuf,10)
val parsedData1 = rd.map{ line =>
  val features = line.tail
  // for(i <- 1 to line.size-1) features(i-1) = line(i)
  /// Vector Dense
  // LabeledPoint(line(0).toInt, Vectors.dense(features))
  /// Vector Sparse
  val Seq = features.zipWithIndex.map(e => (e._2,e._1)).filter(_._2 != 0).unzip
  LabeledPoint(line(0).toInt,Vectors.sparse(features.size,Seq._1.toArray,Seq._2.toArray))
}
val parsedData = parsedData1.randomSplit(Array(0.1,0.9),seed = 11L)(0)
// Evaluate model on training examples and compute training error 
val labelAndPreds = parsedData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = labelAndPreds.filter( r => 
    r._1 != r._2).count.toDouble / parsedData.count
    println("Training Error = " + trainErr)
    println("Precision = " + (1-trainErr) )
-----------------------------------------------------------------------------

-----------------------------------------------------------------------------

/// MTrick QMD res Test
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD 
import org.apache.spark.mllib.util.MLUtils
import org.apache.log4j.Level
import org.apache.log4j.Logger
val data_path = "hdfs:///user/hadoop/suanec/res.data"//"hdfs:///user/hadoop/suanec/QMDEncodeRes.data"
val label_path = "hdfs:///user/hadoop/suanec/Train.label"
val test_data_path = "hdfs:///user/hadoop/suanec/QMDEncodeTestRes.data"
val test_label_path = "hdfs:///user/hadoop/suanec/Test.label"
Logger.getRootLogger.setLevel(Level.ERROR)
/// Load train data
val data = sc.textFile(data_path)
val label = sc.textFile(label_path)
val datas = data.collect
val labels = label.collect
val dist = datas.size - labels.size -1
val data1 = datas.tail.dropRight(dist)
val vec = new Array[Array[Double]](labels.size)

/// Load test data
val test_data = sc.textFile(test_data_path)
val test_label = sc.textFile(test_label_path)
val test_datas = test_data.collect
val test_labels = test_label.collect
val test_dist = test_datas.size - test_labels.size -1
val test_data1 = test_datas.tail.dropRight(test_dist)
val test_vec = new Array[Array[Double]](test_labels.size)

/// Generate the train vec
val dataLength = datas.head.split(',').size
for( i <- 0 until vec.size){
  val tmp = new Array[Double](dataLength+1)
  val splits = data1(i).split(',')
  val dataDouble = new Array[Double](dataLength)
  tmp(0) = (labels(i).toDouble -1)
  for( j <- 0 until splits.size ){
    dataDouble(j) = splits(j).toDouble
  }
  dataDouble.copyToArray(tmp,1)
  vec(i) = tmp
}

/// Generate the test vec
val test_dataLength = test_datas.head.split(',').size
for( i <- 0 until test_vec.size){
  val tmp = new Array[Double](test_dataLength+1)
  val splits = test_data1(i).split(',')
  val dataDouble = new Array[Double](test_dataLength)
  tmp(0) = (test_labels(i).toDouble -1)
  for( j <- 0 until splits.size ){
    dataDouble(j) = splits(j).toDouble
  }
  dataDouble.copyToArray(tmp,1)
  test_vec(i) = tmp
}

/// Generate the train RDD
val rd = sc.parallelize(vec,10)
val parsedData1 = rd.map{ line =>
  val features = line.tail
  /// Vector Dense
  // LabeledPoint(line(0).toInt, Vectors.dense(features))
  /// Vector Sparse
  val Seq = features.zipWithIndex.map(e => (e._2,e._1)).filter(_._2 != 0).unzip
  LabeledPoint(line(0).toInt,Vectors.sparse(features.size,Seq._1.toArray,Seq._2.toArray))
}
val parsedData = parsedData1

/// Generate the test RDD
val test_rd = sc.parallelize(test_vec,10)
val testData1 = test_rd.map{ line =>
  val features = line.tail
  val Seq = features.zipWithIndex.unzip
  LabeledPoint(line(0).toInt,Vectors.sparse(features.size,Seq._2.toArray,Seq._1.toArray))
}
val testParsedData = testData1

/// train the model
/*
val randomSplits = parsedData1.randomSplit(Array(0.7,0.3),seed = 123L)
val parsedData = randomSplits(0)
val testData = randomSplits(1)
val labelAndPreds = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = labelAndPreds.filter( r => r._1 != r._2).count.toDouble / parsedData.count
    println("Training Error = " + trainErr)
    val trainPrecision = 1- trainErr
    println("parsedData Precision = " + trainPrecision )
    */
val numIterations = 200 
val model = SVMWithSGD.train(parsedData, numIterations)  

// Evaluate model on training examples and compute training error 
val labelAndPreds = parsedData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = labelAndPreds.filter( r => r._1 != r._2).count.toDouble / parsedData.count
    println("Training Error = " + trainErr)
    val trainPrecision = 1- trainErr
    println("parsedData Precision = " + trainPrecision )

val testLabelAndPreds = testParsedData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = testLabelAndPreds.filter( r => r._1 != r._2).count.toDouble / parsedData.count
    println("Training Error = " + trainErr)
    val trainPrecision = 1- trainErr
    println("test Precision = " + trainPrecision )

-----------------------------------------------------------------------------

-----------------------------------------------------------------------------

/// MTrick QMD res Test MultiTest
/// resArray[Double] = (res_train_0.5,res_train_0.6,res_train_0.7,res_train_0.8,res_train_0.9)
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD 
import org.apache.spark.mllib.util.MLUtils
import org.apache.log4j.Level
import org.apache.log4j.Logger

Logger.getRootLogger.setLevel(Level.ERROR)


/// Load train data
val data_path = "hdfs:///user/hadoop/suanec/res.data"//"hdfs:///user/hadoop/suanec/QMDEncodeRes.data"
val label_path = "hdfs:///user/hadoop/suanec/Train.label"
val data = sc.textFile(data_path)
val label = sc.textFile(label_path)
val datas = data.collect
val labels = label.collect
val dist = datas.size - labels.size -1
val data1 = datas.tail.dropRight(dist)
val vec = new Array[Array[Double]](labels.size)

/// Generate the train vec
val dataLength = datas.head.split(',').size
for( i <- 0 until vec.size){
  val tmp = new Array[Double](dataLength+1)
  val splits = data1(i).split(',')
  val dataDouble = new Array[Double](dataLength)
  tmp(0) = (labels(i).toDouble -1)
  for( j <- 0 until splits.size ){
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
  val Seq = features.zipWithIndex.map(e => (e._2,e._1)).filter(_._2 != 0).unzip
  LabeledPoint(line(0).toInt,Vectors.sparse(features.size,Seq._1.toArray,Seq._2.toArray))
}

/// train the model
val randomSplits = 
    new Array[
        Array[org.apache.spark.rdd.RDD[
        org.apache.spark.mllib.regression.LabeledPoint]]](5)
/// resArray[Double] = (res_train_0.5,res_train_0.6,res_train_0.7,res_train_0.8,res_train_0.9)
val resArray = new Array[Double](5)
for(i <- 0 until 5){
  val fraq = 0.5 + i.toDouble/10
  randomSplits(i) = parsedData1.randomSplit(Array(fraq.toDouble,(1-fraq.toDouble)),seed = (9*i + 43).toLong )
}
for(i <- 0 until 5){
  val parsedData = randomSplits(i)(0)
  val testData = randomSplits(i)(1)

  Logger.getRootLogger.setLevel(Level.INFO)
  val numIterations = 200 
  val model = SVMWithSGD.train(parsedData, numIterations)  
  Logger.getRootLogger.setLevel(Level.ERROR)

  // Evaluate model on training examples and compute training error 
  val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction) } 
  val trainErr = labelAndPreds.filter( r => r._1 != r._2).count.toDouble / testData.count
  resArray(i) = trainErr

}

resArray.foreach(println)

-----------------------------------------------------------------------------

-----------------------------------------------------------------------------

/// 文件操作企图转换数据格式，任务失败
val writer = new PrintWriter(new File("parsedLibTrain.data"))
/// for(i <- 0 to strArr.size-1){
for(i <- 0 to  strArr.size-1){ 
  writer.write(labels(i))
  writer.write(' ')
  writer.write(strArr(i))
  writer.write("\n")
}
val data = MLUtils.loadLibSVMFile(sc, "file:///home/hadoop/suanec/data/MTrick/parsedLibTrain.data",7270)
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

val numIterations = 100
val model = SVMWithSGD.train(training, numIterations)
-----------------------------------------------------------------------------

-----------------------------------------------------------------------------


import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils



var initStartTime = System.currentTimeMillis()
// Load training data in LIBSVM format.
val data = MLUtils.loadLibSVMFile(sc, "file:///home/hadoop/suanec/suae/workspace/data/mllib/sample_libsvm_data.txt")
//100*692
//features'size is 692

// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

// Run training algorithm to build the model
val numIterations = 100
val model = SVMWithSGD.train(training, numIterations)

// Clear the default threshold.
model.clearThreshold()

// Compute raw scores on the test set.
val scoreAndLabels = test.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}
var arr = scoreAndLabels.collect
var i = 0
var count = 0.0 
for (i <- 0 to arr.size-1){
  var one = (if( arr(i)._1 > 0 ) 1.0 else 0.0 )
  var tuple = (one, arr(i)._2)
  arr(i) = tuple
  if( tuple._1 == tuple._2 ) count += 1
}
println("The Precision is:"+count/arr.size)

var initEndTime = System.currentTimeMillis()
scala.math.ceil((initEndTime - initStartTime).toDouble / 1000)
-----------------------------------------------------------------------------

-----------------------------------------------------------------------------


// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()
//MulticlassMetrics

println("Area under ROC = " + auROC)

// Save and load model
model.save(sc, "myModelPath")
val sameModel = SVMModel.load(sc, "myModelPath")
-----------------------------------------------------------------------------

-----------------------------------------------------------------------------


import org.apache.spark.SparkContext 
import org.apache.spark.mllib.classification.SVMWithSGD 
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint  
// Load and parse the data file 
val data = sc.textFile("file:///home/hadoop/suanec/suae/workspace/data/mllib/sample_svm_data.txt") 
//322*16
val parsedData = data.map { line =>
    val parts = line.split(' ')
    LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(x => x.toDouble).toArray))
    }  
// Run training algorithm to build the model 
val numIterations = 20 
val model = SVMWithSGD.train(parsedData, numIterations)  
// Evaluate model on training examples and compute training error 
val labelAndPreds = parsedData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = labelAndPreds.filter( r => 
    r._1 != r._2).count.toDouble / parsedData.count
    println("Training Error = " + trainErr)

file:///home/hadoop/suanec/suae/workspace/data/mllib/sample_svm_data.txt
file:///home/hadoop/suanec/suae/workspace/SelfAETest/result.txt
file:///home/hadoop/suanec/data/MTrick
Train.data  1977*7270 185803 
Train.label
Test.data
Test.label

-----------------------------------------------------------------------------

-----------------------------------------------------------------------------

//确认map顺序不变。


/// 测试所有label的合法性
val label = parsedData.map(_.label)
val labels = label.collect
for(i <- labels) if(i.toInt != 1 || i.toInt != 0)println(i)
/// output all elements of labels
for(i <- 0 to labels.size-1) if(i%70 == 69)println(labels(i))else print(labels(i))

-----------------------------------------------------------------------------

-----------------------------------------------------------------------------

def parseData( data_path:String, label_path:String ) : RDD[LabeledPoint] = {
  val data = sc.textFile(data_path)
  val label = sc.textFile(label_path)
  val datas = data.collect
  val labels = label.collect
  // val arr = new ArrayBuffer[ArrayBuffer[String]]
  val strArr = new Array[String] (datas(datas.size-1).split(',')(1).toInt)
  val vecArr = new Array[Double] (7270)
  val vecBuf = new ArrayBuffer[Array[Double]]
  val dataBuf = new ArrayBuffer[Array[Double]]
  var colNum = 0
  var str = ""
  var rowNum = 0
  for(i <- 0 to datas.size-1){
    val splits = datas(i).split(',')
    if(rowNum != splits(1).toInt){
      strArr(splits(1).toInt-1) = splits(0) + ':' + splits(2)
      rowNum = splits(1).toInt
    }else{
      strArr(rowNum-1) += " " + splits(0) + ':' + splits(2)
    }
  }/// 得到Index：Feature字符串数组strArr

  for(i <- 0 to strArr.size-1){
    for(j <- 0 to vecArr.size-1) vecArr(j) = 0.0
    val splits = strArr(i).split(' ')
    for(j <- 0 to splits.size-1){
      val parts = splits(j).split(':')
      vecArr(parts(0).toInt-1) = parts(1).toDouble
    }
    val diff = vecArr.max - vecArr.min
    for(j <- 0 to vecArr.size-1){ 
      vecArr(j) = vecArr(j)/diff
    }
    vecBuf += vecArr
  }/// 得到特征向量数组。
  for(i <- 0 to vecBuf.size*2/3){
    // dataBuf += Array(labels(i).toDouble,vecBuf(i))
    dataBuf += new Array[Double]((vecBuf(i).size+1))
    dataBuf(i)(0) = labels(i).toDouble - 1.0
    print(dataBuf(i)(0) + "  ")
    vecBuf(i).copyToArray(dataBuf(i),1,vecBuf(i).size )
  }/// 得到训练数据
  val rd = sc.parallelize(dataBuf,10)
  val parsedData = rd.map{ line =>
    val features = line.tail
    // for(i <- 1 to line.size-1) features(i-1) = line(i)
    /// Vector Dense
    // LabeledPoint(line(0).toInt, Vectors.dense(features))
    /// Vector Sparse
    val Seq = features.zipWithIndex.map(e => (e._2,e._1)).filter(_._2 != 0).unzip
    LabeledPoint(line(0).toInt,Vectors.sparse(features.size,Seq._1.toArray,Seq._2.toArray))
  }

}// parseData

-----------------------------------------------------------------------------

-----------------------------------------------------------------------------

/// random shuffle
import java.util.Random
import java.util.LinkedList
val ll = new LinkedList[String]
for(i <- 'a' to 'z'){
  val ti = i.toInt%7
  for(j <- ti to 0 by -1){
    ll.add(i.toString)
  }
}
val rate = 0.4
val idx = (ll.size * rate).toInt
val lleft = ll.size - idx
val rand0 = new Random()
val rand = new Random(((rate * idx * rand0.nextInt * rand0.nextInt * ll.size / rate )%397).toLong)
for(i <- 0 to idx){
  val ti = ll.get(i)
  var index = idx
  val rd = rand.nextInt % lleft
  if(rd < 0){
    index = index + rd * (-1)
  }else{
    index = index + rd
  }
  ll.set(i,ll.get(index))
  ll.set(index,ti)
}
/// random shuffle array
import scala.collection.mutable.ArrayBuffer
import java.util.Random
val ab = new ArrayBuffer[String]
for( i <- 'a' to 'z'){
  val ti = i.toInt%7
  for(j <- ti to 0 by -1){
    ab+= i.toString
  }
}
var rate = 0.4
val idx = (ab.size * rate).toInt
val abLeft = ab.size - idx
val rand = new Random(System.currentTimeMillis)
for(i <- 0 to idx){
  val si = ab(i)
  var rd = rand.nextInt % abLeft
  if(rd < 0) rd = rd * -1
  val index = idx + rd
  ab(i) = ab(index)
  ab(index) = si
}

-----------------------------------------------------------------------------

-----------------------------------------------------------------------------

/// open close lamp
val startTime = System.currentTimeMillis
val arr = new Array[Boolean] (101)
for( i <- 1 until arr.size ){
  var idx = 1
  var j = 1
  while(idx <= 100 && j <= 100 ){
    idx = i * j
    j = j+1
    arr(idx) = !arr(idx)
    print(idx)
    print('\t')
    idx = i * j
  }
  println
}
arr.filter(_ == true).size
val endTime = System.currentTimeMillis
endTime - startTime
/// quickly open close lamp
val startTime = System.currentTimeMillis
val arr = new Array[Boolean] (101)
for( i <- 1 to 10 ){
  arr(i*i) = true
}
val result = arr.tail.filter( _ == true ).size
val endTime = System.currentTimeMillis
endTime - startTime

-----------------------------------------------------------------------------
 