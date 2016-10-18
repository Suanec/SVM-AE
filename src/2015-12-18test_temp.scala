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
type SDV = BDV[Double]
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
  import java.util.concurrent.ThreadLocalRandom

  
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
    // val rand = new Random(seed)
    val bdm_rand = BDM.tabulate[Double](rows, cols)((_, _) => ThreadLocalRandom.current().nextDouble(min,max))
    val bdm1 = bdm_rand * (max - min) + min 
    bdm1
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

  def encode(_wIn : SDM, _docs : Sdocs ) : SDM = {
    var z = new SDM(_wIn.rows, 0)
    _docs.map{
      doc =>
      println(doc.head._2)
      if(z.size == 0) z = encode(_wIn,doc)
      else{
        val t = z
        z = BDM.horzcat(t,encode(_wIn,doc))
      }
    }
    // val res = breeze.numerics.sigmoid(z)
    val res = z
    res
  }/// def encode(_wIn,_docs)

  /**
   *  
   *  
   *  @author suanec
   */
  def encode(_wIn : SDM, _row : SDM): SDM = {
    val z = _wIn * _row
    val hideOut = breeze.numerics.sigmoid(z)
    hideOut
  }///  def encode(_wIn : SDM, _row : SDM): Sdoc = {

  // def encode(_wIn : SDM, _data : Sdata ) : RDD[]

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
    val tm  = BDM.zeros[Double](_wOut.rows,_wOut.cols)
    tm += _wOut
    (0 until _doc.size).map{
      i =>
      val col = _doc(i)._1 -1
      val value = _doc(i)._3
      val v = tm(col,::) * _hideOut
      val v1 = breeze.numerics.sigmoid(v)
      res(i) = (col + 1, row + 1, v1.t.data.head )
    }
    res
  }

  /**
   *  reconstruction error square
   *  @author suanec
   */
  def calcSquareError( _inMat : SDM, _outMat : SDM ) : Double = {
    val minus = _outMat - _inMat
    val sqMinus = pow(minus,2.0)
    val res = sum(sqMinus)
    res
  }

  def calcErrDoc( _inDoc : Sdoc, _outDoc : Sdoc ) : Double = {
    require(_inDoc.size == _outDoc.size , "bad data , ==from boae.calcSqrErrDoc")
    val in = _inDoc.sorted
    val out = _outDoc.sorted
    var sum = 0d
    (0 until in.size).map{
      i =>
      require(in(i)._1 == out(i)._1,"bad data , == from boae.calcSqrErrDoc.map")
      sum += (out(i)._3 - in(i)._3)
    }
    sum * sum
  }

  /**
   *  
   *  softmax 
   *  @author suanec
   */
  def calcSoftmax( _w2 : SDM, _hideOut : SDM, _label : SDV ) : Double = {
    
  }
    // AA = errors.*outputs.*(1-outputs);
    // BB = hiddenvalues1.*(1-hiddenvalues1);
    // CC = labelvalues.*(1-labelvalues);
    // DD = hiddenvalues.*(1-hiddenvalues);

  /**
   *  DKL(P||Q) = sum(P(i)ln(Pi/Qi))
   *  return a Double
   *  @author suanec
   */
  def calcKL( _ps : SDV, _pt : SDV ) : Double = {
    val div = _ps :/ _pt
    val lnDiv = breeze.numerics.log(div)
    val toSum = _ps :* lnDiv
    val res = breeze.linalg.sum(toSum)
    res
  }/// def calcKL

  /**
   *  symmetrized DKL(ps,pt)
   *  DKL(ps||pt) + DKL(pt||ps)
   *  return a Double
   *  @author suanec
   */
  def calcSymKL( _sHiddenMat : SDM, _tHiddenMat : SDM ) : Double = {
    val ps0 : SDV = sum(_sHiddenMat(*,::))
    val ps1 : SDV = ps0 :* (1.0 / ps0.size)
    val ps : SDV = ps0 :/ ps1
    val pt0 : SDV = sum(_tHiddenMat(*,::))
    val pt1 : SDV = pt0 :* (1.0 / pt0.size)
    val pt : SDV = pt0 :/ pt1
    val stDKL = calcKL(ps,pt)
    val tsDKL = calcKL(pt,ps)
    val res = stDKL + tsDKL
    res 
  }/// def calcSymKL

  /**
   *  regularization
   *  omiga = ||W1||^2 + ||b1||^2 + ||W2||^2 + ||b2||^2 +
   *          ||W11||^2 + ||b11||^2 + ||W22||^2 + ||b22||^2 
   *  return a Double
   *  @author suanec
   */
  def calcRegular( _w : Array[SDM], _b : Array[SDM] ) :Double = {
    var res = 0.0
    _w.map( w => res += sum(w) )
    _b.map( b => res += sum(b) )
    res
  }/// calcRegular

}/// boae

val ds = readMTrick()
val dsv = ds.values
val docs = dsv.collect 
val doc = docs.head
val train = docs.map(k => k._2)
val w = boae.InitWeight(Array(ds.cols,80,ds.cols))
val hide = boae.encode(w.head,train)
val out = boae.decode(w.last,hide,train)
// val labelz = boae.encode(w.last,hide)
// val hide1 = boae.decode(w.)




val sma = new Array[Array[Double]] (1000)
for(i <- 0 until 1000 - 1){
  sma(i) = ma(i,::).t.toDenseMatrix.toArray
}
sma(999) = ma(881,::).t.toDenseMatrix.toArray
val smb = new Array[Array[Double]] (1000)
(0 until 1000).map{
  i =>
  smb(i) = new Array[Double] (1000)
  sma(999-i).copyToArray(smb(i))
}
val smc = new Array[Array[Double]] (1000)
(0 until 1000).map( i => smc(i) = new Array[Double] (1000) )
val s = System.currentTimeMillis
for(i <- 0 until sma.size){
  var sum = 0d
  for(j <- 0 until smb.head.size){
    for(k <- 0 until sma(i).size){
      sum += sma(i)(k) * smb(k)(j)
    }
    smc(i)(j) = sum
    sum = 0
  }
}
val e = System.currentTimeMillis
e - s


val data = BDM.eye[Double](8)
val train = new Sdocs(8)
(0 until 8).map{
  i =>
  train(i) = Array((i+1,i+1,1d))
}
val aew = InitWeight(Array(8,3,8))
(0 until 5000).map{
  j =>
  if(j < 300 |j > 4800) println("-----------------" + j + "-----------------")
  train.map( i=> NNBP(aew,i))
  val enc = train.map{
    i => 
    encode(aew.head,i).toArray
  }.toArray
  if(j < 300 |j > 4800) 
  enc.map{
    x =>
    x.foreach( i => print( i + "\t"))
    println
  }
}
train.map{
  i =>
  val ii = decode(aew.last,encode(aew.head,i),i)
  (i,"\n",ii,'\n')
}


def MulDoc(
  _w : SDM,
  _doc : Sdoc) : SDM = {
  val tSDM = new SDM(_w.rows,1)
  (0 until _w.rows).map{
    i =>
    (0 until _doc.size).map{
      j =>
      val rNum = _doc(j)._1 -1
      tSDM(i,0) += _w(i,rNum) * _doc(j)._3
    }
    tSDM(i,0) += _w(i,-1)
  }
  tSDM
}

/// 8维单位阵的稀疏表示
val train = new Sdocs(8)
(0 until 8).map{
  i =>
  val ab = new ArrayBuffer[Sword]
  (0 until 8).map{
    j =>
    ab += ((j+1,i+1,0d))
  }
  train(i) = ab.toArray
  train(i)(i) = (i+1,i+1,1d)
  // ab.clear 
}

/// 8维稠密阵的稀疏表示
val train3 = new Sdocs(8)
(0 until 8).map{
  i =>
  train3(i) = new Sdoc (8)
  (0 until 8).map{
    j =>
    train3(i)(j) = (j+1,i+1,util.Random.nextInt(10)/10d)
  }
}
val train = train3
val aew = InitWeight(Array(8,3,8))
(0 until 5000).map{
  j =>
  if(j < 300 |j > 4800) println("-----------------" + j + "-----------------")
  train.map( i=> NNBP(aew,i))
  val enc = train.map{
    i => 
    encode(aew.head,i).toArray
  }.toArray
  if(j < 300 |j > 4800) 
  enc.map{
    x =>
    x.foreach( i => print( i + "\t"))
    println
  }
}
/// 观察输入输出值
train.map{
  i =>
  val ii = decode(aew.last,encode(aew.head,i),i).map( x => (x._1,x._2,ceil(x._3 * 10)/10d))
  (i,"\n     ",ii,'\n','\n')
}
/// 验证calcMinusDoc
train.map{
  i =>
  val ii = decode(aew.last,encode(aew.head,i),i).map( x => (x._1,x._2,ceil(x._3 * 10)/10d))
  (i,"\n     ",ii,"\n     ",calcMinusDoc(i,ii),'\n','\n')
}
/// 验证calcLoss
  val vt = train.head
  val vtt = decode(aew.last,encode(aew.head,vt),vt)
  (vt,vtt,calcLoss(vt,vtt))
/// 验证calcErrorOut
val veoRes = calcErrorOut(vt,vtt)
val o = vtt.map( x => x._3 )
val oo = o.map( x => x * (1-x) )
val m = calcMinusDoc(vt,vtt).map(x => x._3)
(0 until m.size).map( i => m(i) * oo(i) - veoRes(i)._3 )
/// 验证calcErrorHidden
val vtho = encode(aew.head,vt)
val vehRes = calcErrorHidden(veoRes,vtho,aew.last)
val dh = vtho :* (1d - vtho)
val veh = BDM.zeros[Double](vtho.rows,1)
(0 until vtho.rows).map{
  i =>
  (0 until ceoRes.size).map{
    j =>
    veh(i,0) += ceoRes(j)._3 * aew.last(j,i)
  }
}
val veh = new SDM(vtho.rows,1)
(0 until veh.rows).map{
  i =>
  (0 until veoRes.size).map{
    j =>
    val k = veoRes(j)._1 -1
    val v = veoRes(j)._3
    veh(i,0) += v * aew.last(k,i)
  }
}
(veh :* (vtho :* (1d - vtho))) - calcErrorHidden(veoRes,vtho,aew.last)
// (0 until t2.size).map{
//   i => 
//   val k = t2(i)._1 -1
//   val v = t2(i)._3
//   (0 until t1.rows).map{
//     j => 
//     t3(j,0) += v * t(k,j)
//   }
// }
// t3 :*= t1 :* (1d - t1)
// t3 - calcErrorHidden(t2,t1,t)

(veh :* dh) - vehRes

val t = new SDM(2,2,Array(2d,1d,1d,2d))
val t1 = new SDM(2,1,Array(6d,2d))
val t2 = new Sdoc(2)
t2(0) = (1,1,14d)
t2(1) = (2,1,10d)
val t3 = new SDM(t1.rows,1)
(0 until t2.size).map{
  i => 
  val k = t2(i)._1 -1
  val v = t2(i)._3
  (0 until t1.rows).map{
    j => 
    t3(j,0) += v * t(k,j)
  }
}
t3 :*= t1 :* (1d - t1)
t3 - calcErrorHidden(t2,t1,t)

val vaew = new Array[SDM](2)
val vaewH = new SDM(2,3,Array(2d,5d,3d,1d,7d,7d))
vaew(0) = vaewH 
val vaewL = new SDM(2,3,Array(7d,3d,4d,9d,7d,7d))
vaew(1) = vaewL
/// input
val t = new SDM(2,1,Array(6d,2d))
/// delta hide
val t1 = new SDM(2,1,Array(7d,3d))
/// delta out Doc
val t2 = new Sdoc(2)
t2(0) = (1,1,5d)
t2(1) = (2,1,4d)
/// hide out
val t4 = new SDM(2,1,Array(18d,32d))
/// input doc
val t3 = new Sdoc(2)
t3(0) = (1,1,6d)
t3(1) = (2,1,2d)
val vaewr = updateWB(vaew(0),vaew(1),t1,t2,t4,t3,2d)
val ti = vaewr._1.copy
val to = vaewr._2.copy
val vaew = new Array[SDM](2)
val vaewH = new SDM(2,3,Array(2d,5d,3d,1d,7d,7d))
vaew(0) = vaewH 
val vaewL = new SDM(2,3,Array(7d,3d,4d,9d,7d,7d))
vaew(1) = vaewL
val pti = vaew.head.copy
val pto = vaew.last.copy
(ti - pti) / 2d
(pti - ti) / 2d
(pto - to) / 2d



/// 8维验证阵的稀疏表示
val train = Array(Array((1,1,0.1), (2,1,0.5), (3,1,0.8), (4,1,0.7), (5,1,0.8), (6,1,0.8), (7,1,0.0), (8,1,0.8)), Array((1,2,0.7), (2,2,0.8), (3,2,0.6), (4,2,0.1), (5,2,0.4), (6,2,0.8), (7,2,0.7), (8,2,0.0)), Array((1,3,0.8), (2,3,0.6), (3,3,0.4), (4,3,0.1), (5,3,0.2), (6,3,0.4), (7,3,0.3), (8,3,0.2)), Array((1,4,0.6), (2,4,0.5), (3,4,0.2), (4,4,0.1), (5,4,0.0), (6,4,0.5), (7,4,0.3), (8,4,0.9)), Array((1,5,0.5), (2,5,0.0), (3,5,0.1), (4,5,0.3), (5,5,0.0), (6,5,0.6), (7,5,0.0), (8,5,0.9)), Array((1,6,0.5), (2,6,0.9), (3,6,0.7), (4,6,0.7), (5,6,0.0), (6,6,0.7), (7,6,0.8), (8,6,0.3)), Array((1,7,0.2), (2,7,0.3), (3,7,0.5), (4,7,0.8), (5,7,0.9), (6,7,0.1), (7,7,0.0), (8,7,0.4)), Array((1,8,0.1), (2,8,0.7), (3,8,0.7), (4,8,0.1), (5,8,0.2), (6,8,0.9), (7,8,0.5), (8,8,0.0)))
val aewHead = Array(-0.02749883843582579, 0.3885659693044735, 0.12338758351069523, -0.49328892524752843, 0.044969602622286, 0.4745563106029106, 0.38694445472650685, -0.38372880601611337, -0.062390538892880154, -0.25334788575774025, -0.30328334430017545, 0.17846390397614842, -0.350372270654728, -0.2115447060418082, -0.24859061504374413, 0.16905491186420063, -0.4071069940604469, 0.30770514703192475, 0.06600088084827294, -0.09382450497286898, -0.08245441151794675, -0.4256991219095714, 0.48958698715211046, -0.20666930040210252, -0.1837388602590141, -0.029011345924522036, -0.026620962434521944)
val aewLast = Array(-0.2784042410193389, -0.43474429155673244, -0.04101086891670236, 0.15602472157148994, -0.3631093163523623, 0.4985954840794409, 0.30088178790775966, -0.4624777859078858, -0.29111299467350793, 0.33210147278455915, -0.04564976169482615, 0.49835025432941116, 1.4614625626463429E-5, 0.2541163148213885, -0.3470670934779829, 0.1870377967317276, -0.47379917589736753, -0.0173737815065379, -0.176288714981932, 0.3472489070257262, 0.1253878462316631, -0.4482917083342266, 0.15690820304674835, 0.4845595333517968, -0.39446130663959855, -0.06500027173880008, -0.16134311210773566, 0.38345525443324147, 0.23075609784562723, -0.26013818655508947, 0.18574088362651908, -0.12332034876961295)
val aew = InitWeight(Array(8,3,8))
aew(0) = new SDM(3,9,aewHead)
aew(1) = new SDM(8,4,aewLast)
val vt = Array((1,1,0.1), (2,1,0.5), (3,1,0.8), (4,1,0.7), (5,1,0.8), (6,1,0.8), (7,1,0.0), (8,1,0.8))
val vtt = Array((1,1,0.30078886802814037), (2,1,0.47699612396134006), (3,1,0.4288588404462507), (4,1,0.689800941335419), (5,1,0.5462324762637724), (6,1,0.4359127392510418), (7,1,0.5601141475386238), (8,1,0.5156652894700111))
