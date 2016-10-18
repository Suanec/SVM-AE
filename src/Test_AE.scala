package tests

import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.util.MLUtils
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
// import NN.NeuralNet
// import util.RandSampleData



object Test_AE {

  def main(args: Array[String]) {
    //1 构建Spark对象
    val conf = new SparkConf().setAppName("AEtest")
                              //.setMaster("spark://cloud76:7077")
                              .setMaster("local")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    val data_path = "file:///home/hadoop/suanec/suae/workspace/data/mllib/sample_libsvm_data.txt"
    val examples = MLUtils.loadLibSVMFile(sc, data_path).cache()
    // val train_d1 = examples.map { f =>
    //   LabeledPoint(f.label, Vectors.dense(f.features.toArray))
    // }
    var train_data1 = data.map{ f=>
      val label = f.label
      var features = f.features.toArray
      val dist = (features.max - features.min) * 1.0
      for( i <- 0 to features.size -1 ){
        features(i) = features(i) / dist
      }
      ( BDM( label ), BDM( features ) )
    }
     val opts = Array(100.0, 200.0, 0.0)
     val train_d = train_d1.map(f => (BDM(f.label), BDM(f.features.toArray)))
       // val opts = Array(100.0, 200.0, 0.0)
       // val train_d = train_d1.map(f => (BDM((f.label, f.label * 0.5 + 2.0)), BDM(f.features.toArray)))
       // val numExamples = train_d.count()
       // println(s"numExamples = $numExamples.")

    //*****************************例1（基于经典优化算法测试函数随机生成样本）*****************************// 
    //2 随机生成测试数据
    // 随机数生成
    //3 设置训练参数，建立模型
    // opts:迭代步长，迭代次数，交叉验证比例
    val opts = Array(100.0, 20.0, 0.2)
    train_d.cache
    val numExamples = train_d.count()
    println(s"numExamples = $numExamples.")
    val NNmodel = new NeuralNet().
      setSize(Array(692, 300, 692)).
      setLayer(3).
      setActivation_function("tanh_opt").
      setLearningRate(2.0).
      setScaling_learningRate(1.0).
      setWeightPenaltyL2(0.0).
      setNonSparsityPenalty(0.0).
      setSparsityTarget(0.05).
      setInputZeroMaskedFraction(0.0).
      setDropoutFraction(0.0).
      setOutput_function("sigm").
      NNtrain(train_d, opts)

    //4 模型测试
    val NNforecast = NNmodel.predict(train_d)
    val NNerror = NNmodel.Loss(NNforecast)
    println(s"NNerror = $NNerror.")
    val printf1 = NNforecast.map(f => (f.label.data(0), f.predict_label.data(0))).take(200)
    println("预测结果——实际值：预测值：误差")
    for (i <- 0 until printf1.length)
      println(printf1(i)._1 + "\t" + printf1(i)._2 + "\t" + (printf1(i)._2 - printf1(i)._1))
    println("权重W{1}")
    val tmpw0 = NNmodel.weights(0)
    for (i <- 0 to tmpw0.rows - 1) {
      for (j <- 0 to tmpw0.cols - 1) {
        print(tmpw0(i, j) + "\t")
      }
      println()
    }
    println("权重W{2}")
    val tmpw1 = NNmodel.weights(1)
    for (i <- 0 to tmpw1.rows - 1) {
      for (j <- 0 to tmpw1.cols - 1) {
        print(tmpw1(i, j) + "\t")
      }
      println()
    }

  }
}

