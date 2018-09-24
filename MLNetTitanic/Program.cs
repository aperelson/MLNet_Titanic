using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Models;

namespace MLNetTitanic
{
    class Program
    {
        const string TrainDataPath = "train_data.csv";
        const string TestDataPath = "test_data.csv";
        const string RealTestDataPath = "real_test.csv";

        public class PassengerData
        {
            [Column(ordinal: "1", name: "Label")]
            public bool Survived;

            [Column(ordinal: "2")]
            public float Pclass;

            [Column(ordinal: "4")]
            public string Sex;

            [Column(ordinal: "5")]
            public float Age;

            [Column(ordinal: "6")]
            public float SibSp;

            [Column(ordinal: "7")]
            public float Parch;

            [Column(ordinal: "8")]
            public string Ticket;

            [Column(ordinal: "9")]
            public float Fare;

            [Column(ordinal: "10")]
            public string Cabin;

            [Column(ordinal: "11")]
            public string Embarked;
        }

        // SurvivePrediction is the result returned from prediction operations
        public class TitanicPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool IsSurvived;

            [ColumnName("Score")]
            public float ScoredScore;
        }


        static void Main(string[] args)
        {
            var testDataLoader = new TextLoader(TestDataPath).CreateFrom<PassengerData>(useHeader: true, separator: ',');

            PredictionModel<PassengerData, TitanicPrediction> model_FastTreeBinaryClassifier = Train_FastTreeBinaryClassifier();
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model_FastTreeBinaryClassifier, testDataLoader);

            Console.WriteLine("");
            Console.WriteLine($"Train_FastTreeBinaryClassifier");
            Console.WriteLine($"Accuracy: {metrics.Accuracy} F1 Score: {metrics.F1Score}");
            Console.WriteLine($"True Positive: {metrics.ConfusionMatrix[0, 0]} False Positive: {metrics.ConfusionMatrix[0, 1]}");
            Console.WriteLine($"False Negative: {metrics.ConfusionMatrix[1, 0]} True Negative: {metrics.ConfusionMatrix[1, 1]}");


            List<PassengerData> testThis = new List<PassengerData>();

            using (var rd = new StreamReader(RealTestDataPath))
            {
                rd.ReadLine();

                while (!rd.EndOfStream)
                {
                    var splits = rd.ReadLine().Split(',');

                    testThis.Add(new PassengerData
                    {
                        Pclass = (float)Convert.ToDecimal(splits[1].ToString()),
                        Sex = splits[4],
                        Age = (float)Convert.ToDecimal(splits[5].ToString()),

                    });
                }
            }

                        /*
        [Column(ordinal: "6")]
        public float SibSp;

        [Column(ordinal: "7")]
        public float Parch;

        [Column(ordinal: "8")]
        public string Ticket;

        [Column(ordinal: "9")]
        public float Fare;

        [Column(ordinal: "10")]
        public string Cabin;

        [Column(ordinal: "11")]
        public string Embarked;

        column1.Add(splits[0]);
                    column2.Add(splits[1]);
                }
            }



            var realTestDataLoader = new TextLoader(RealTestDataPath).CreateFrom<PassengerData>(useHeader: true, separator: ',');

            var temp2 = realTestDataLoader

            PassengerData data = new PassengerData
            {
                Age = 73,
                Sex = "Female"
            };
            */
            //var temp2 = CollectionDataSource.Create(realTestDataLoader);


            //IEnumerable<TitanicPrediction> prediction = model_FastTreeBinaryClassifier.Predict();

            //Console.WriteLine("prediction: " + prediction.IsSurvived);
            Console.ReadLine();
        }

        private static PredictionModel<PassengerData, TitanicPrediction> Train_FastTreeBinaryClassifier()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(TrainDataPath).CreateFrom<PassengerData>(useHeader: true, separator: ','),

                new CategoricalOneHotVectorizer("Sex", "Embarked"),

                new ColumnConcatenator("Features",
                    "Age",
                    "Pclass",
                    "SibSp",
                    "Parch",
                    "Sex",
                    "Embarked"),

                new FastTreeBinaryClassifier() { NumTrees = 100 }
            };

            return pipeline.Train<PassengerData, TitanicPrediction>();
        }

        private static PredictionModel<PassengerData, TitanicPrediction> Train_FastForestBinaryClassifier()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(TrainDataPath).CreateFrom<PassengerData>(useHeader: true, separator: ','),

                new CategoricalOneHotVectorizer("Sex", "Embarked"),

                new ColumnConcatenator("Features",
                    "Age",
                    "Pclass",
                    "SibSp",
                    "Parch",
                    "Sex",
                    "Embarked"),

                new FastForestBinaryClassifier () { NumTrees = 100 }
            };

            return pipeline.Train<PassengerData, TitanicPrediction>();
        }

        private static PredictionModel<PassengerData, TitanicPrediction> Train_LogisticRegressionClassifier()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(TrainDataPath).CreateFrom<PassengerData>(useHeader: true, separator: ','),

                new CategoricalOneHotVectorizer("Sex", "Embarked"),

                new ColumnConcatenator("Features",
                    "Age",
                    "Pclass",
                    "SibSp",
                    "Parch",
                    "Sex",
                    "Embarked"),

                new LogisticRegressionBinaryClassifier () 
            };

            return pipeline.Train<PassengerData, TitanicPrediction>();
        }
    }
}
