import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


case class NNConfig(
  var size: Array[Int] = Array(10,2,10),
  var layer: Int = 3,
  var learningRate: Double = 0.1f,
  var iteration : Int = 10,
  var lr : Double = 0.1f,
  activation_function: String = "sigmoid",
  momentum: Double = 0.5f,
  scaling_learningRate: Double = 0.5f,
  weightPenaltyL2: Double = 0.5f,
  nonSparsityPenalty: Double = 0.5f,
  sparsityTarget: Double = 0.2f,
  inputZeroMaskedFraction: Double = 0.8f,
  dropoutFraction: Double = 0.3f,
  testing: Double = 0.1f,
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

}/// NNConfig  case class

object RUtil extends Serializable {

  type Svector = Array[Double]
  type Sword = (Int,Int,Double)
  type SwordBuf = ArrayBuffer[Sword]
  type Sdoc = Array[Sword]
  type Sdocs = ArrayBuffer[Sdoc]
  type Sdata = (Int, Sdoc)
  type SDataSet = (Int, Int, RDD[Sdata])
  case class DataStats(var rows : Int, var cols : Int, var values : RDD[Sdata]) extends Serializable {
    def setRows( row : Int ) = {
      require(row >= 0, "error row ! == From dataStats.setRows")
      this.rows = row
    }
    def setCols( col : Int ) = {
      require( col >= 0, "error col ! == From dataStats.setCols")
      this.cols = col
    }
    def setValues( value : RDD[Sdata] ) = {
      this.values = value
      setRows(this.values.count.toInt)
      setCols(this.values.map{
        doc => 
          doc._2.last._1
        }.collect.max.toInt
      )
    }
    def this() = this(0,0,null)
    def this(value : RDD[Sdata]) = {
      this(0,0,null)
      setValues(value)

    }
  }/// DataStats case class

  def Sigmoid( row : Svector ) : Svector = {
    row.map{
      x => (1 / (1 + Math.exp(-1 * x))).toDouble
    }
  }/// Sigmoid
 
  def softmax(arr : Sdoc) = {
    var sum = 0.0
    var empty = true
    val res = new Sdoc(0)
    arr.foreach{
       i =>
       if(i._3 <= 709) sum += Math.exp(i._3).toDouble
       else empty = false
    }
    if(empty)  (0 until arr.size).map( i => (arr(i)._1, arr(i)._2, (Math.exp(arr(i)._3) /sum).toDouble) ).toArray
    else res.toArray
  }/// softmax

  def readMTrick( sc : SparkContext,
    isNorm : Boolean = true, 
    file : String = "hdfs:///user/hadoop/suanec/Train.data",
    minPartitions : Int = 64 ) : DataStats = {
    val data = sc.textFile( file, minPartitions ).map{
      line =>
        val splits = line.split(',')
        (splits(0).toInt, splits(1).toInt, splits(2).toDouble)
    }  
    val cols = data.max._1.toInt
    val rows = data.max()(Ordering.by[(Int,Int,Double),Int](_._2))._2.toInt

    val docs = data.map( x => (x._2,(x._1, x._2, x._3)) ).groupByKey 
    val normedDocs = docs.map{
      doc => 
        val words = if(isNorm){ softmax(doc._2.toArray) } else doc._2.toArray
        (doc._1, words)
    }.filter( doc => doc._2.size != 0 )
    new DataStats(rows, cols, normedDocs)
  }/// readMTricks return dataStats

}/// RUtil