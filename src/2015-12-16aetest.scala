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
  import scala.math._
  import java.util.concurrent.ThreadLocalRandom


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
    tanh => Btanh,
    sigmoid
  }

  type SDM = BDM[Double]
  type SDV = BDV[Double]
  type Sword = (Int,Int,Double)
  type Sdoc = Array[Sword]
  type Sdocs = Array[Sdoc]

  type Svector = Array[Double]
  type Sword = (Int,Int,Double)
  type SwordBuf = ArrayBuffer[Sword]
  type Sdoc = Array[Sword]
  type Sdocs = Array[Sdoc]
  type Sdata = (Int, Sdoc)
  type SDataSet = (Int, Int, RDD[Sdata])
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
    // val rand = new Random(seed)
    val bdm_rand = BDM.tabulate[Double](rows, cols)((_, _) => ThreadLocalRandom.current().nextDouble(min,max))
    bdm_rand 
  }///   def RandM(

  /**
   * 将数组按行填充矩阵。
   * 
   */
  def arrFillMatByRow(
    _row : Int,
    _col : Int,
    _data: Array[Double]) : SDM = {
    require(_row*_col <= _data.size, "target matrix too large, _data not enough!! == From boae.arrRowMat")
    BDM.tabulate[Double](_row,_col)((i,j) => _data( i*_col + j))
  }
  /**
   * 自编码权值初始化函数
   * _size : 网络大小
   * max: 上限，min: 下限，seed: 随机种子
   * @author suanec
   */
  def InitWeight(
    _size  : Array[Int],
    _max   : Double = 0.5,
    _min   : Double = -0.5,
    _seed  : Long = 201512082245L) : Array[SDM] = {

    val res = new Array[SDM](2)
    res(0) = RandM(_size(1),_size(0)+1,_max,_min,_seed)
    res(1) = RandM(_size(2),_size(1)+1,_max,_min,_seed)
    res
  }///   def InitWeight(

  def MulDoc(
    _mat : SDM,
    _doc : Sdoc) : SDM = {
    val row = _doc.head._2 -1
    val tm  = _mat.copy
    _doc.map{
      word =>
      val col = word._1 -1
      val value = word._3
      (0 until _mat.rows).map( i => tm(i, col) *= value )
    }
    val res = sum(tm(*,::)).toDenseMatrix.t
    res
  }

  def encode(_wIn: SDM, _doc : Sdoc ): SDM ={
    val z = MulDoc(_wIn,_doc)
    val hideOut = breeze.numerics.sigmoid(z)
    hideOut//.toDenseMatrix.t
  }///   def encode(_wIn: SDM, _doc : Sdoc ): SDM ={

  def decode(_wOut : SDM, _hideOut: SDM, _doc : Sdoc ) : Sdoc = {
    val cList = _doc.map( x => x._1 ).toArray.sorted
    val rNum = _doc.head._2 
    val bHidden = BDM.vertcat( _hideOut, new SDM(1,1,Array(1d)))
    val z = _wOut * bHidden
    val a1 = breeze.numerics.sigmoid(z)
    val res = new Sdoc(cList.size)
    (0 until res.size).map( i => res(i) = (cList(i),rNum,a1(cList(i)-1,0)))
    res
  }///   def decode(_wOut : SDM, _hideOut: SDM ) : SDM = {

  def calcMinusDoc( _inDoc : Sdoc, _outDoc : Sdoc ) : Sdoc = {
    require(_inDoc.size == _outDoc.size , "bad data , ==from boae.calcMinusDoc")
    val in = _inDoc.sorted
    val out = _outDoc.sorted
    val rNum = out.head._2
    val res = new Sdoc(in.size)
    (0 until res.size).map{
      i => 
      res(i) = (in(i)._1,rNum,(out(i)._3 - in(i)._3))
    }
    res
  }

  def calcLoss( _inDoc : Sdoc, _outDoc : Sdoc ) : Double = {
    val minusDoc = calcMinusDoc(_inDoc,_outDoc)
    val sumArr = minusDoc.map{
      word =>
      word._3 * word._3
    }.toArray
    val loss = sumArr.sum / sumArr.size.toDouble
    loss
  }

  def NNFF( _w : Array[SDM], _doc : Sdoc ) : (SDM,Sdoc) = {
    val hideOut = encode(_w.head,_doc)
    val docOut = decode(_w.last,hideOut,_doc)
    (hideOut,docOut)
  }/// NNFF

  def calcErrorOut( _inDoc : Sdoc, _outDoc :Sdoc ) : Sdoc = {
    val res = calcMinusDoc( _inDoc, _outDoc )
    val rNum = res.head._2
    (0 until res.size).map{
      i => 
      res(i) = (res(i)._1,rNum,
        (res(i)._3 * _outDoc(i)._3 * (1 - _outDoc(i)._3)) )
    }
    res
  }

  def calcErrorHidden( _errOut : Sdoc, _hideOut : SDM,  _wOut : SDM ) : SDM = {
    // val wOutT = _wOut.t
    // val res = MulDoc( wOutT, _errOut )
    val errorHidden = new SDM(_hideOut.rows,1)
    for(i <- 0 until _errOut.size){
      val k = _errOut(i)._1 -1
      val v = _errOut(i)._3
      for(j <- 0 until _hideOut.size){
        errorHidden(j,0) += _wOut(k,j) * v        
      }
    }
    val h = _hideOut :* ( 1.0 - _hideOut )
    errorHidden :*= h
    errorHidden
  }

  def updateWB(
    _wIn : SDM, 
    _wOut : SDM, 
    _errHidden : SDM, 
    _errOut : Sdoc,
    _hideOut : SDM,    
    _doc : Sdoc,
    _lr : Double ) : (SDM,SDM) = {
    val dIn = _wIn.copy * 0d
    // dIn(::,-1) += _wIn(::,-1)
    val dOut = _wOut.copy * 0d
    // dOut(::,-1) += _wOut(::,-1)
    for( i <- 0 until _doc.size ){
      val k = _doc(i)._1 -1
      val v = _doc(i)._3
      val vOut = _errOut(i)._3
      for( j <- 0 until _hideOut.size ){
        dIn(j,k) += v * _errHidden(j,0)
        dOut(k,j) += vOut * _hideOut(j,0)
        dIn(j,-1) = _errHidden(j,0)
      }
      dOut(k,-1) += vOut
    }
    _wIn -= (dIn * _lr)
    _wOut -= (dOut * _lr)
    (_wIn,_wOut)
  }

  def NNBP( _w : Array[SDM], _inDoc : Sdoc ) : (SDM,SDM) = {
  // def NNFF( _w : Array[SDM], _doc : Sdoc ) : (SDM,Sdoc) = {
    val (hideOut,docOut) = NNFF(_w,_inDoc)

  // def calcErrorOut( _inDoc : Sdoc, _outDoc :Sdoc ) : Sdoc = {
    val loss = calcLoss(_inDoc,docOut)
    val errDoc = calcErrorOut(_inDoc,docOut)
    println(s"---- Line : ${errDoc.head._2}, Loss : $loss ----")

  // def calcErrorHidden( _errOut : Sdoc, _hideOut : SDM,  _wOut : SDM ) : SDM = {
    val errorHidden = calcErrorHidden( errDoc, hideOut, _w.last )

  // def updateWB(
  //   _wIn : SDM, 
  //   _wOut : SDM, 
  //   _errHidden : SDM, 
  //   _errOut : Sdoc,
  //   _hideOut : SDM,    
  //   _doc : Sdoc,
  //   _lr : Double ) : (SDM,SDM) = {
    val nW = updateWB( _w.head, _w.last, errorHidden, errDoc, hideOut, _inDoc, 0.00001 )
    (nW._1,nW._2)
  }
 
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
}/// readMTrick return dataStats


// val ds = readMTrick()
// val docs = ds.values.collect 
// val aew = InitWeight(Array(ds.cols,100,ds.cols))
// val doc = docs.head._2
// NNBP(aew,doc)
// docs.map( doc => NNBP(aew,doc._2))
val enc = docs.map{
  doc => 
  val rDoc = doc._2
  val rNum = rDoc.head._2
  val rEnc = encode(aew.head,rDoc)
  (rNum,rEnc.toArray)
}



import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint

val labels = sc.textFile("hdfs:///user/hadoop/suanec/Train.label").map(x => x.toInt).collect

val labeledPoint = enc.map( x => LabeledPoint( labels(x._1-1)-1, Vectors.dense(x._2)))
val rdd = sc.parallelize(labeledPoint) 
// val svmData = sc.parallelize(points)
val svmSplits = rdd.randomSplit(Array(0.9,0.1),System.currentTimeMillis)
val svmTrain = svmSplits.head
val svmTest = svmSplits.last
val numIter = 20
val model = SVMWithSGD.train(svmTrain,numIter)

val labelAndPreds = svmTest.collect.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = labelAndPreds.filter( r => r._1 != r._2).size.toDouble / svmTest.count
    println("Training Error = " + trainErr)
    val trainPrecision = 1- trainErr
    println("Precision = " + trainPrecision )
///// precision 0.5
