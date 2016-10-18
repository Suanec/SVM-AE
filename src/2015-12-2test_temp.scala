
class Smatrix(
    private var rows : Int, 
    private var cols : Int, 
    private var values : Array[Svector], 
    private var rand : Random
    ) extends Serializable {
  def this() = this(0,0,null,new Random(134L))
  
  def this( values : Array[Svector] ){
    this( values.size.toInt, 0, values, null)
    this.rows = values.size.toInt
    values.map{
      i =>
        if( cols < i.size.toInt ) cols = i.size.toInt
    }
    this.values = new Array[Svector](rows)      
    (0 until values.size).map{
      i => 
        this.values(i) = new Svector(cols)
        values(i).copyToArray(this.values(i),0)
    }
  }///  def Smatrix( values : Array[Svector] )

  def this( rows : Int, cols : Int ){
    this( rows, cols, null, new Random(System.currentTimeMillis()) )
    this.values = new Array[Svector](rows)
    (0 until this.values.size).map{
      i => 
        this.values(i) = new Array[Double](cols)
    } 
  }/// Smatrix zero
  
  def this( rows : Int, cols : Int, seed : Long ) = {
    this( rows, cols, null, new Random(seed))
    this.values = new Array[Svector](rows)
    (0 until this.values.size).map{
      i => 
        this.values(i) = new Array[Double](cols)
        (0 until this.values(i).size).map{
          j => this.values(i)(j) = (rand.nextDouble()%0.8f) + 0.1f - 0.3f /// -0.2 ~ 0.6
        }
    }
  }/// Smatrix random

  def mulRow( row : Svector ) : Svector = {
    require( row.size == this.cols , "Dimision fault! --From Smatrix.mulRow")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until this.cols).map{
          j =>
            aij += this.values(i)(j) * row(j) 
        }
        res(i) = aij
    }
    res
  }/// mulRow
  
  def netMulRow( row : Svector ) : Svector = {
    require( (row.size + 1) == this.cols , "Dimision fault! --From Smatrix.netMulRow")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until this.cols -1 ).map{
          j =>
            aij += this.values(i)(j) * row(j) 
        }
        res(i) = aij + this.values(i).last
    }
    res
  }/// netMulRow
  
  def mulDoc( doc : Sdoc ) : Svector = {
    doc.sortWith(_._1 < _._1)
    require( this.cols >= doc.last._1, "Dimision fault! --From Smatrix.mulDoc")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until doc.size).map{
          j => 
            val tCol = doc(j)._1 - 1
            aij += this.values(i)(tCol) * doc(j)._3
        }
        res(i) = aij
    }
    res
  }/// mulDoc
  
  def netMulDoc( doc : Sdoc ) : Svector = {
    doc.sortWith(_._1 < _._1)
    require( this.cols+1 >= doc.last._1 + 1, "Dimision fault! --From Smatrix.netMulDoc")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until doc.size).map{
          j => 
            val tCol = doc(j)._1 - 1
            aij += this.values(i)(tCol) * doc(j)._3
        }
        res(i) = aij + this.values(i).last
    }
    res
  }/// mulDoc
  
  def size() : (Int,Int) = (rows,cols)
  
  def value() : Array[Svector] = this.values
  
  def getRow( row : Int ) : Svector = {
    require( row <= this.rows && row > 0 ,"Rows over!!  == From Smatrix.getRow.")
    this.values(row-1)
  }
  
  def getElem( row : Int, col : Int ) : Double = {
    require( row <= this.rows && col <= this.cols && row > 0 && col > 0, "No Index in Smatrix!! == From Smatrix.getElem.")
    this.values(row-1)(col-1)
  }
  
  def updateElem( row : Int, col : Int, value : Double ) : Double = {
    require( row <= this.rows && col <= this.cols && row > 0 && col > 0, "No Index in Smatrix!! == From Smatrix.updateElem.")
    this.values(row-1)(col-1) = value
    this.values(row-1)(col-1)
  }
  
  
}/// Smatrix

class SAutoEncoder(
  private var Ws : Array[Smatrix], 
  private var Bs : Svector, 
  private var config : NNConfig
  
  ) extends Serializable {

  def getConf() : NNConfig = this.config
  
  def getW() : Array[Smatrix] = this.Ws
  
  def setLr(lr : Double) = this.config.setLr(lr)

  def this(config : NNConfig){
    this(null,null,config)
    InitWeight(config)
  }/// def this(config : NNConfig) 
  
  def InitWeight(config : NNConfig){
    val arr = config.size
    this.Ws = new Array[Smatrix](arr.size -1)
    this.Bs = new Svector (arr.size)
    var t = (0 until arr.size-1 ).map{
      i =>
        this.Ws(i) = new Smatrix( arr(i+1).toInt, (arr(i)+1).toInt, System.currentTimeMillis() )
        this.Bs(i) = 1.0 
    }
  }/// InitWeight !

  def Init(){
    if(this.config != null) InitWeight(this.config)
  }
  
  def getEncode( inData : Sdocs, resPath : String ) = {
    val res = inData.map{
      inDoc =>
        (inDoc.head._2,encode(inDoc))
    }
    import scala.io.Source
    res    
  }
  
  def SGDtrain(inData : Sdocs){
    val res = inData.map { 
      inDoc => 
        (0 until this.config.iteration).map{
          j => 
            print(s"Iteration : $j ,\t")
            NNBP(inDoc)
        } 
    }
  }/// train
  
  def encode( row : Svector ) : Svector = Sigmoid(this.Ws.head.netMulRow(row)) /// encode!
  
  def encode( doc : Sdoc ) : Svector = Sigmoid(this.Ws.head.netMulDoc(doc)) /// encode!
  
  def decode( row : Svector ) : Svector = Sigmoid(this.Ws.last.netMulRow(row) ) /// decode!
  
  def decode( doc : Sdoc, row : Svector ) : Sdoc = {
    val docCols = doc.map( _._1 )
    val numCols = docCols.size
    val docID = doc.head._2
    require( docID == doc.last._2 , "doc Data Error !! == From SAutoEncoder.decode(doc,row)")
    val w1 = this.Ws.last.value()
    val sigIn = (0 until numCols).map{
      i =>
        var sum = 0.0 
        (0 until row.size).map{
          j =>
            sum += w1(docCols(i)-1)(j) * row(j)
        }
        sum += w1(docCols(i)-1).last
        sum
    }.toArray
    val sigOut = Sigmoid(sigIn)
    val res = (0 until numCols).map( i => (docCols(i),docID,sigOut(i)) ).toArray
    res
  } ///decode doc!
  
  def NNFF( doc : Sdoc ) : ( Sdoc, Svector ) ={
    val hiddenOut = encode(doc)
    val endOut = decode(doc,hiddenOut)
    (endOut, hiddenOut)
  } /// NNFF
  
  def NNBP( inDoc : Sdoc ){
    val (endOut,hiddenOut) = NNFF( inDoc )
    val errOutDoc = CalcErr( inDoc, endOut )
    val errHid = CalcErrHidden(errOutDoc, endOut, hiddenOut)
    UpdateW( inDoc, hiddenOut, errHid, errOutDoc)
    
  }/// NNBP
  
  def CalcErr( target : Sdoc, out : Sdoc) : Sdoc = {
    var sum = 0.0 
    /// (t-o)*(1-o)*o
    require( target.size == out.size, "Bad Dimision!!  == From SAutoEncoder.CalcErr")
    target.sortBy { x => x._1 }
    out.sortBy { x => x._1 }
    val minus = (0 until out.size).map{ 
      i =>
        require( target(i)._1 == out(i)._1, "Bad column!! == From SAutoEncoder.CalcErr")
        (target(i)._3 - out(i)._3)
    }
    val loss = minus.map{ x => x * x }.sum / minus.size 
    println(s"row : ${out(1)._2} ,\tloss : $loss , \thalfLoss : ${loss/2}")
    val sigdiver = out.map{
      x =>
        /// outputValue => oVal
        val oVal = x._3
        val res = oVal * (1 - oVal)
        res
    }
    val errOutDoc = (0 until out.size).map{
      i =>
        (out(i)._1, out(i)._2, minus(i) * sigdiver(i))
    }.toArray
    errOutDoc
  }/// CalcErr
  
  def CalcErrHidden( errOutDoc : Sdoc, out : Sdoc, hiddenOut : Svector ) : Svector ={
    var sum = 0.0
    /// Error out => eo
    /// sum(k) -> eh(k) += eo(h) * w(h,k)
    /// Error Hidden => eh; z = hiddenOut(i)
    /// eh(i) = eh(i) * z * (1-z)
    val w = this.Ws.last
    val err = errOutDoc.sortBy { x => x._1 }.map{ i => i._3 }.toArray
    val errLen = err.size
    val hideLen = hiddenOut.size
    val errHid = new Svector(hideLen)
    (0 until errLen).map{
      i =>
        (0 until hideLen).map{
          j =>
            errHid(j) += err(i) * w.getElem(out(i)._1, (j+1).toInt)
        }
    }
    
    (0 until hideLen).map{
      i =>
        val z = hiddenOut(i)
        errHid(i) *= z * ( 1 - z )
    }
    errHid
  }/// CalcErrHiden
  
  def UpdateW( inDoc : Sdoc, hideOut : Svector, errHid : Svector, errOutDoc : Sdoc){
    val lr = this.config.learningRate
    val hiddenSize = hideOut.size
    val outSize = errOutDoc.size
    /// output layer : deltaW(j,i) = lr * eo(j) * ho(i)
    
    // ( 0 until outSize ).map{
    //   j =>
    //   (0 until hiddenSize).map{
    //     i =>
    //       val k = errOutDoc(j)._1
    //       this.Ws.last.updateElem(k, (i+1).toInt, 
    //           ( this.Ws.last.getElem(k, (i+1).toInt) + (lr * errOutDoc(j)._3 * hideOut(i)).toDouble ))
    //       this.Ws.head.updateElem((i+1).toInt, k, 
    //           ( this.Ws.head.getElem((i+1).toInt, k) + (lr * errHid(i) * inDoc(j)._3 ).toDouble ))
    //   }
    // }

    val alpha = 1
    val beta = 2
    val bIndex = this.config.size.head
    (0 until errHid.size).map{
      i => 
      this.Ws.head.updateElem( (i+1), this.Ws.head.getRow(i+1).size, 
        ( this.Ws.head.getRow(i+1).last + (lr*(errHid(i) * alpha - this.Ws.head.getRow(i+1).last * beta ))))
    }/// update Bin
    ( 0 until outSize ).map{
      j => 
      val k = errOutDoc(j)._1 
      this.Ws.last.updateElem( k, this.Ws.last.getRow(k).size,
        ( this.Ws.last.getRow(k).last + (lr*(errOutDoc(j)._3 * alpha - this.Ws.last.getRow(k).last * beta))))
      (0 until hiddenSize).map{
        i =>
        this.Ws.head.updateElem( (i+1), k,
          ( this.Ws.head.getElem((i+1), k) + (lr*(errHid(i) * inDoc(j)._3 - this.Ws.head.getElem((i+1), k) * beta))))
        this.Ws.last.updateElem( k, (i+1),
          ( this.Ws.last.getElem(k, (i+1)) + (lr*(errOutDoc(j)._3 * hideOut(i) - this.Ws.last.getElem(k,(i+1)) * beta))))
      }
    }

    // for (j <- 0 to config.K - 1) {
    //   Bin(j) += config.stepLength * (errorHidden(j) * alpha - Bin(j) * beta)
    // }
    // for (i <- 0 to Xin.length - 1) {
    //   Bout(Xin(i)._1) += config.stepLength * (errorOut(i) * alpha + Bout(Xin(i)._1) * beta)
    //   for (j <- 0 to config.K - 1) {
    //     Win(Xin(i)._1)(j) -= config.stepLength * (errorHidden(j) * Xin(i)._2 * alpha + Win(Xin(i)._1)(j) * beta)
    //     Wout(Xin(i)._1)(j) -= config.stepLength * (errorOut(i) * Xhidden(j) * alpha + Win(Xin(i)._1)(j) * beta)
    //   }
    // }
  }/// UpdateW
}



import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint


val ds = readMTrick()
val dsv = ds.values.map(x => x._2).collect
val conf = new NNConfig
conf.setSize(Array(ds.cols,900,ds.cols))
conf.setLr(0.01)
val ae = new SAutoEncoder(conf)
ae.SGDtrain(dsv)
val enc = ae.getEncode(dsv,"")
val labels = sc.textFile("hdfs:///user/hadoop/suanec/Train.label").map( x=> x.toInt -1 ).collect
val lp = enc.map( x => LabeledPoint(labels(x._1-1),Vectors.dense(x._2))).toArray
val pd = sc.parallelize(lp)
val sp = pd.randomSplit(Array(0.9,0.1),11874937395L)
val train = sp.head
val test = sp.last
val model = SVMWithSGD.train(train,80)
val lp = test.collect.map( x => (x.label, model.predict(x.features)))
val precision = lp.filter(x => x._1 == x._2).size.toDouble/lp.size







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

type Svector = Array[Double]
type Sword = (Int,Int,Double)
type SwordBuf = ArrayBuffer[Sword]
type Sdoc = Array[Sword]
type Sdocs = Array[Sdoc]
type SdocBuf = ArrayBuffer[Sdoc]
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

def readMTrick( isNorm : Boolean = true, 
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

val labels = sc.textFile("hdfs:///user/hadoop/suanec/Train.label",1).map(s => s.toInt -1).collect
val ds = readMTrick()
val dsv = ds.values
val dsvr = dsv.map( x => ( x._1, x._2.map(z => (z._1,z._3)).unzip))
val dslp = dsvr.map( x => LabeledPoint( labels(x._1 -1), Vectors.sparse( ds.cols+1, x._2._1.toArray, x._2._2.toArray) ) )
val sp = dslp.randomSplit(Array(0.9,0.1),System.currentTimeMillis)
val train = sp.head
val test = sp.last
val model = SVMWithSGD.train(train,20)
val precision = test.map( x => (x.label, model.predict(x.features) ) ).filter(x => x._1 == x._2).count.toDouble/test.count.toDouble
/// precision: Double = 0.912621359223301



val dsn = readMTrick(false)
val dsnv = dsn.values
val dsnr = dsnv.map( x => ( x._1, x._2.map(z => (z._1,z._3)).unzip))
val dsnlp = dsnr.map( x => LabeledPoint(labels(x._1-1),Vectors.sparse( ds.cols +1, x._2._1.toArray, x._2._2.toArray) ) )
val spn = dsnlp.randomSplit(Array(0.9,0.1),System.currentTimeMillis)
val trainn = spn.head
val testn = spn.last
val modeln = SVMWithSGD.train(trainn,20)
val lpn = testn.map( x => (x.label, modeln.predict(x.features) ) )
val precisionn = lpn.filter( x => x._1 == x._2 ).count.toDouble/lpn.count.toDouble
/// precisionn: Double = 0.994475138121547


class Smatrix(
    private var rows : Int, 
    private var cols : Int, 
    private var values : Array[Svector], 
    private var rand : Random
    ) extends java.io.Serializable {
  def this() = this(0,0,null,new Random(134L))
  
  def this( values : Array[Svector] ){
    this( values.size.toInt, 0, values, null)
    this.rows = values.size.toInt
    values.map{
      i =>
        if( cols < i.size.toInt ) cols = i.size.toInt
    }
    this.values = new Array[Svector](rows)      
    (0 until values.size).map{
      i => 
        this.values(i) = new Svector(cols)
        values(i).copyToArray(this.values(i),0)
    }
  }///  def Smatrix( values : Array[Svector] )

  def this( rows : Int, cols : Int ){
    this( rows, cols, null, new Random(System.currentTimeMillis()) )
    this.values = new Array[Svector](rows)
    (0 until this.values.size).map{
      i => 
        this.values(i) = new Array[Double](cols)
    } 
  }/// Smatrix zero
  
  def this( rows : Int, cols : Int, seed : Long ) = {
    this( rows, cols, null, new Random(seed))
    this.values = new Array[Svector](rows)
    (0 until this.values.size).map{
      i => 
        this.values(i) = new Array[Double](cols)
        (0 until this.values(i).size).map{
          j => this.values(i)(j) = (rand.nextDouble()%0.8f) + 0.1f - 0.3f /// -0.2 ~ 0.6
        }
    }
  }/// Smatrix random

  def mulRow( row : Svector ) : Svector = {
    require( row.size == this.cols , "Dimision fault! --From Smatrix.mulRow")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until this.cols).map{
          j =>
            aij += this.values(i)(j) * row(j) 
        }
        res(i) = aij
    }
    res
  }/// mulRow
  
  def netMulRow( row : Svector ) : Svector = {
    require( (row.size + 1) == this.cols , "Dimision fault! --From Smatrix.netMulRow")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until this.cols -1 ).map{
          j =>
            aij += this.values(i)(j) * row(j) 
        }
        res(i) = aij + this.values(i).last
    }
    res
  }/// netMulRow
  
  def mulDoc( doc : Sdoc ) : Svector = {
    doc.sortWith(_._1 < _._1)
    require( this.cols >= doc.last._1, "Dimision fault! --From Smatrix.mulDoc")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until doc.size).map{
          j => 
            val tCol = doc(j)._1 - 1
            aij += this.values(i)(tCol) * doc(j)._3
        }
        res(i) = aij
    }
    res
  }/// mulDoc
  
  def netMulDoc( doc : Sdoc ) : Svector = {
    doc.sortWith(_._1 < _._1)
    require( this.cols+1 >= doc.last._1 + 1, "Dimision fault! --From Smatrix.netMulDoc")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until doc.size).map{
          j => 
            val tCol = doc(j)._1 - 1
            aij += this.values(i)(tCol) * doc(j)._3
        }
        res(i) = aij + this.values(i).last
    }
    res
  }/// mulDoc
  
  def size() : (Int,Int) = (rows,cols)
  
  def value() : Array[Svector] = this.values
  
  def getRow( row : Int ) : Svector = {
    require( row <= this.rows && row > 0 ,"Rows over!!  == From Smatrix.getRow.")
    this.values(row-1)
  }
  
  def getElem( row : Int, col : Int ) : Double = {
    require( row <= this.rows && col <= this.cols && row > 0 && col > 0, "No Index in Smatrix!! == From Smatrix.getElem.")
    this.values(row-1)(col-1)
  }
  
  def updateElem( row : Int, col : Int, value : Double ) : Double = {
    require( row <= this.rows && col <= this.cols && row > 0 && col > 0, "No Index in Smatrix!! == From Smatrix.updateElem.")
    this.values(row-1)(col-1) = value
    this.values(row-1)(col-1)
  }
  
  
}/// Smatrix


------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

object osm extends Serializable{
  var rows : Int = 0
  var cols : Int = 0
  var values : Array[Svector] = null
  var rand : Random = null
  def Init( _value : Array[Svector]) : Array[Svector] = {
    rows = _value.size.toInt
    cols = _value.map( i => i.size.toInt ).max
    values = new Array[Svector](rows)
    (0 until values.size).map{
      i => 
      values(i) = new Svector(cols)
      _value(i).copyToArray(values(i),0)
    }
    values
  }/// osm Init

  def Init( _rows : Int, _cols : Int ) : Array[Svector] = {
    rows = _rows
    cols = _cols
    values = new Array[Svector](rows)
    (0 until values.size).map{
      i => 
        values(i) = new Array[Double](cols)
    } 
    values
  }/// osm zero
  
  def Init( _rows : Int, _cols : Int, _seed : Long ) : Array[Svector] = {
    Init(_rows, _cols)
    rand = new Random(_seed)
    (0 until this.values.size).map{
      i => 
        (0 until this.values(i).size).map{
          j => this.values(i)(j) = (rand.nextDouble()%0.8f) + 0.1f - 0.3f /// -0.2 ~ 0.6
        }
    }
    values
  }/// osm random

  def mulRow( _w : Array[Svector], _row : Svector ) : Svector = {
    require( _row.size == _w.head.size , "Dimision fault! --From osm.mulRow")
    require( _row.size == _w.last.size , "Data Dimision fault! --From osm.mulRow")
    val res = new Svector(_w.size)
    val t = (0 until res.size).map{
      i =>
        var aij = 0.0 
        (0 until _w(i).size).map{
          j =>
            aij += _w(i)(j) * _row(j) 
        }
        res(i) = aij
    }
    res
  }/// osm mulRow

  def netMulRow( _w : Array[Svector], _row : Svector ) : Svector = {
    require( (_row.size+1) == _w.head.size , "Dimision fault! --From osm.mulRow")
    require( (_row.size+1) == _w.last.size , "Data Dimision fault! --From osm.mulRow")
    val res = new Svector(_w.size)
    val t = (0 until _w.size).map{
      i =>
        var aij = 0.0 
        (0 until _w(i).size-1 ).map{
          j =>
            aij += _w(i)(j) * _row(j) 
        }
        res(i) = aij + _w(i).last
    }
    res
  }/// netMulRow
  
}/// osm  
  def mulDoc( doc : Sdoc ) : Svector = {
    doc.sortWith(_._1 < _._1)
    require( this.cols >= doc.last._1, "Dimision fault! --From Smatrix.mulDoc")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until doc.size).map{
          j => 
            val tCol = doc(j)._1 - 1
            aij += this.values(i)(tCol) * doc(j)._3
        }
        res(i) = aij
    }
    res
  }/// mulDoc
  
  def netMulDoc( doc : Sdoc ) : Svector = {
    doc.sortWith(_._1 < _._1)
    require( this.cols+1 >= doc.last._1 + 1, "Dimision fault! --From Smatrix.netMulDoc")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until doc.size).map{
          j => 
            val tCol = doc(j)._1 - 1
            aij += this.values(i)(tCol) * doc(j)._3
        }
        res(i) = aij + this.values(i).last
    }
    res
  }/// mulDoc
  
  def size() : (Int,Int) = (rows,cols)
  
  def value() : Array[Svector] = this.values
  
  def getRow( row : Int ) : Svector = {
    require( row <= this.rows && row > 0 ,"Rows over!!  == From Smatrix.getRow.")
    this.values(row-1)
  }
  
  def getElem( row : Int, col : Int ) : Double = {
    require( row <= this.rows && col <= this.cols && row > 0 && col > 0, "No Index in Smatrix!! == From Smatrix.getElem.")
    this.values(row-1)(col-1)
  }
  
  def updateElem( row : Int, col : Int, value : Double ) : Double = {
    require( row <= this.rows && col <= this.cols && row > 0 && col > 0, "No Index in Smatrix!! == From Smatrix.updateElem.")
    this.values(row-1)(col-1) = value
    this.values(row-1)(col-1)
  }
  
  
}/// Smatrix

