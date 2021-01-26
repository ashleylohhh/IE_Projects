#$ bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.2 pyspark-shell

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.streaming.kafka import TopicAndPartition
from pyspark.sql import Row,SQLContext
import sys
import requests

def aggregate_tags_count(new_values, total_sum):
    return sum(new_values) + (total_sum or 0)

def get_sql_context_instance(spark_context):
    if ('sqlContextSingletonInstance' not in globals()):
        globals()['sqlContextSingletonInstance'] = SQLContext(spark_context)
    return globals()['sqlContextSingletonInstance']

def process_rdd(time, rdd):
    print("----------- %s -----------" % str(time))
    try:
        # Get spark sql singleton context from the current context
        sql_context = get_sql_context_instance(rdd.context)
        # convert the RDD to Row RDD
        row_rdd = rdd.map(lambda w: Row(word=w[0], word_count=w[1]))
        # create a DF from the Row RDD
        words_df = sql_context.createDataFrame(row_rdd)
        # Register the dataframe as table
        words_df.registerTempTable("words")
        # get the top 10 words from the table using SQL and print them
        word_counts_df = sql_context.sql("select word, word_count from words order by word_count desc limit 10")
        word_counts_df.show()
    except:
        e = sys.exc_info()[0]
        print("Error: %s" % e)

# create spark configuration
conf = SparkConf()
conf.setAppName("TwitterStreamApp")

# create spark context with the above configuration
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

# create the Streaming Context from the above spark context with interval size 2 seconds
ssc = StreamingContext(sc, 2)

# setting a checkpoint to allow RDD recovery
ssc.checkpoint("checkpoint")

# read data from the port
tweets = ssc.socketTextStream("localhost", 9009)
#tweets = KafkaUtils.createDirectStream(ssc, topic=['topic'], kafkaParams={"metadata.broker.list":"localhost:9009"})

# split each tweet into words
words = tweets.flatMap(lambda line: line.split(" ")).map(lambda x: (x, 1))

# obtain the Top10 words (not only hashtags) using a moving window of 10 minutes every 30 seconds
#def updateFunc(new_values, last_sum):
    #return sum(new_values) + (last_sum or 0)
wordCounts = words.reduceByKeyAndWindow(lambda x, y: int(x) + int(y), lambda x, y: int(x) - int(y), 600, 30)

wordCounts.pprint()

# do the processing for each RDD generated in each interval
wordCounts.foreachRDD(process_rdd)

# start the streaming computation
ssc.start()

# wait for the streaming to finish
ssc.awaitTermination()