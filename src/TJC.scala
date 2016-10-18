
------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object TCJ{

  case class NNConfig(
    var size: Array[Int] = Array(10,2,10),
    var layer: Int = 3,
    var learningRate: Double = 0.1,
    var iteration : Int = 2,
    var lr : Double = 0.1,
    var alpha : Double = 1,
    var beta : Double = 2,
    activation_function: String = "sigmoid",
    momentum: Double = 0.5,
    scaling_learningRate: Double = 0.5,
    weightPenaltyL2: Double = 0.5,
    nonSparsityPenalty: Double = 0.5,
    sparsityTarget: Double = 0.2,
    inputZeroMaskedFraction: Double = 0.8,
    dropoutFraction: Double = 0.3,
    testing: Double = 0.1,
    output_function: String = "sigmoid") extends Serializable {
    
    def setLr( lr : Double ) = {
      require(lr >= 0, "error lr ! == From NNConfig.setLearningRate")
      this.lr = lr 
      this.learningRate = this.lr
    }
    
    def setLayer( layer : Int ) = {
      require(layer >= 0, "error layer ! == From NNConfig.setLayer")
      this.layer = layer
    }
    
    def setSize( size : Array[Int] ) = {
      this.size = size
      setLayer(size.size)
    }
    
    def setIteration( numIter : Int ) = {
      require( numIter >= 1 , "error numIter ! == From NNConfig.setIteration")
      this.iteration = numIter
    }
    
    def setAlpha( _alpha : Double ) = this.alpha = _alpha
    
    def setBeta( _beta : Double ) = this.beta = _beta
  
  }/// NNConfig  case class

  def main( arg : Array[String] ) = {
    println("val conf = new NNConfig")
    val conf = new NNConfig
    println("val arr = Array(conf,conf,conf,conf,conf,conf)")
    val arr = Array(conf,conf,conf,conf,conf,conf)
    println("val sconf = new SparkConf().setAppName("TJC")")
    val sconf = new SparkConf().setAppName("TJC")
    println("val sc = new SparkContext(sconf)")
    val sc = new SparkContext(sconf)
    println("val rdd = sc.parallelize(arr,3)")
    val rdd = sc.parallelize(arr,3)
    println("println(rdd.count)")
    println(rdd.count)
    val t = rdd.first
    println("t.setLr(999999.999999)")
    t.setLr(999999.999999)
    println(t.toString)
  }



}