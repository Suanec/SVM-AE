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


  type Svector = Array[Double]
  type Sword = (Int,Int,Double)
  type SwordBuf = ArrayBuffer[Sword]
  type Sdoc = Array[Sword]
  type Sdocs = Array[Sdoc]
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
      setCols( this.values.max()(Ordering.by[Sdata,Int]( x => x._2.last._1 ) )._2.last._1.toInt )
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

def readMTrick( isNorm : Boolean = true, 
  file : String = "hdfs:///user/hadoop/suanec/Train.data",
  minPartitions : Int = 4 ) : DataStats = {
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

/**
 *  
 *  
 *  
 */
import breeze.linalg._
import breeze.numerics._
import scala.util.Random
import breeze.linalg.{
  DenseMatrix => BDM,
  DenseVector => BDV,
  sum
}
import breeze.numerics.{
  exp => Bexp,
  tanh => Btanh
}

type SDM = BDM[Double]
type Sword = (Int,Int,Double)
type Sdoc = Array[Sword]
type Sdocs = Array[Sdoc]



val a = new Array[Double](12)
val b = new Array[Double](12)
val rand = new Random(System.currentTimeMillis)
(0 until a.size).map{
  i => 
  a(i) = 0.01 + ((rand.nextDouble/0.1).toInt)/10.0 
  b(i) = 0.01 + ((rand.nextDouble/0.1).toInt)/10.0 
}
val ma = new SDM(4,3,a)
val mb = new SDM(4,3,b)

object boae extends Serializable{
  
  import breeze.numerics._
  import breeze.linalg.{
    DenseMatrix => BDM,
    DenseVector => BDV,
    sum
  }
  import breeze.numerics.{
    exp => Bexp,
    tanh => Btanh
  }
  import scala.util.Random
  
  type SDM = BDM[Double]
  type Sword = (Int,Int,Double)
  type Sdoc = Array[Sword]
  type Sdocs = Array[Sdoc]

  /**
   * 随机生成n2维数据
   * rows 行, cols 列，max 上限，min 下限，seed 随机种子
   * @author suanec
   */
  def RandM(
    rows : Int = 4,
    cols : Int = 3,
    max  : Double = 1.0,
    min  : Double = 0.0,
    seed : Long = 201512081552L): BDM[Double] = {

    val rand = new Random(seed)
    val bdm_rand = BDM.tabulate[Double](rows, cols)((_, _) => rand.nextDouble)
    val bdm1 = bdm_rand * (max - min) + min 
    bdm1
  }///   def RandM(

  /**
   * 自编码权值初始化函数
   * _size : 网络大小
   * max: 上限，min: 下限，seed: 随机种子
   * @author suanec
   */
  def InitWeight(
    _size  : Array[Int]
    _max   : Double = 0.5,
    _min   : Double = -0.5,
    _seed  : Long = 201512082245L) : Array[SDM] = {

    val res = new Array[SDM](2)
    res(0) = RandM(_size(1)+1,_size(0),_max,_min,_seed)
    res(1) = RandM(_size(2)+1,_size(1),_max,_min,_seed)
    res
  }///   def InitWeight(

  /**
   * BDM稀疏运算doc乘法
   * 结果为DenseVector
   * _mat 权值矩阵， _doc 稀疏向量
   */
  def MulDoc(
    _mat : SDM,
    _doc : Sdoc
    ) : SDM = {
    val res = new SDM(_mat.rows,1)
    val row = _doc.head._2 -1
    val tm  = BDM.zeros[Double](_mat.rows,_mat.cols)
    tm += _mat
    _doc.map{
      word =>
      val col = word._1 -1
      val value = word._3
      (0 until _mat.rows).map( i => tm(i, col) *= value )
    }
    (0 until _mat.rows).map{
      i => 
      res(i,0) = sum(tm(i,::))
    }
    res
  }///   def MulDoc(

  /**
   * breeze 提供了sigmoid函数，可直接调用：sigmoid(mat)
   * sigm激活函数
   * X = 1./(1+exp(-P));
   */
  def sigm(_matrix: BDM[Double]): BDM[Double] = {
    val ret = 1.0 / (Bexp(_matrix * (-1.0)) + 1.0)
    ret
  }///   def sigm(_matrix: BDM[Double]): BDM[Double] = {

  /**
   *  boae.encode
   *
   *  @author suanec
   */
  def encode(_wIn: SDM, _doc : Sdoc ): SDM ={
    val z = MulDoc(_wIn,_doc)
    val hideOut = breeze.numerics.sigmoid(z)
    hideOut
  }///   def encode(_wIn: SDM, _doc : Sdoc ): SDM ={

  /**
   *  
   *  
   *  @author suanec
   */
  def encode(_wIn : SDM, _row : SDM): Sdoc = {
    val z = _wIn * _row
    val hideOut = breeze.numerics.sigmoid(z)
    hideOut
  }///  def encode(_wIn : SDM, _row : SDM): Sdoc = {

  /**
   *  
   *  
   *  @author suanec
   */
  def decode(_wOut : SDM, _hideOut: SDM ) : SDM = {
    val z = _wOut * _hideOut
    val a1 = breeze.numerics.sigmoid(z)
    a1
  }///   def decode(_wOut : SDM, _hideOut: SDM ) : SDM = {

  /**
   *  
   *  
   *  @author suanec
   */
  def decode(_wOut : SDM, _hideOut : SDM, _doc: Sdoc) : Sdoc = {
    val res = new Sdoc(_doc.size)
    val row = _doc.head._2 -1
    val tm  = BDM.zeros[Double](_mat.rows,_mat.cols)
    tm += _mat
    (0 until _doc.size).map{
      i =>
      val col = _doc(i)._1 -1
      val value = _doc(i)._3
      val v = tm(col,::) * _hideOut
      res(i) = (col + 1, row + 1, v)
    }
    res
  }
  /**
   *  
   *  
   *  @author suanec
   */
  /**
   *  
   *  
   *  @author suanec
   */
  /**
   *  
   *  
   *  @author suanec
   */





}/// boae