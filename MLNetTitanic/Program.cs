using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using CsvHelper;
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
        const string ResultsDataPath = "results.csv";

        public class PassengerData
        {
            [Column(ordinal: "0")]
            public float PassengerId;

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

            
            PredictionModel<PassengerData, TitanicPrediction> model = Train_FastTreeBinaryClassifier();
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testDataLoader);

            Console.WriteLine("");
            Console.WriteLine($"Train_FastTreeBinaryClassifier");
            Console.WriteLine($"Accuracy: {metrics.Accuracy} F1 Score: {metrics.F1Score}");
            Console.WriteLine($"True Positive: {metrics.ConfusionMatrix[0, 0]} False Positive: {metrics.ConfusionMatrix[0, 1]}");
            Console.WriteLine($"False Negative: {metrics.ConfusionMatrix[1, 0]} True Negative: {metrics.ConfusionMatrix[1, 1]}");

            /*
            PredictionModel<PassengerData, TitanicPrediction> model = Train_LogisticRegressionClassifier();
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testDataLoader);

            Console.WriteLine("");
            Console.WriteLine($"Train_LogisticRegressionClassifier");
            Console.WriteLine($"Accuracy: {metrics.Accuracy} F1 Score: {metrics.F1Score}");
            Console.WriteLine($"True Positive: {metrics.ConfusionMatrix[0, 0]} False Positive: {metrics.ConfusionMatrix[0, 1]}");
            Console.WriteLine($"False Negative: {metrics.ConfusionMatrix[1, 0]} True Negative: {metrics.ConfusionMatrix[1, 1]}");
            */


            List<PassengerData> testThis = ReadPassengerData(RealTestDataPath);

            //List<PassengerData> testThis = new List<PassengerData>();
            /*
            using (var rd = new StreamReader(RealTestDataPath))
            {
                rd.ReadLine();

                while (!rd.EndOfStream)
                {
                    var splits = rd.ReadLine().Split(',');

                    testThis.Add(new PassengerData
                    {
                        PassengerId = (float)Convert.ToDecimal(splits[0].ToString()),
                        Pclass = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(splits[1]) ? "0" : splits[1].ToString()),
                        Sex = splits[4],
                        Age = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(splits[5]) ? "0" : splits[5].ToString()),
                        SibSp = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(splits[6]) ? "0" : splits[6].ToString()),
                        Parch = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(splits[7]) ? "0" : splits[7].ToString()),
                        Ticket = splits[8],
                        Fare = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(splits[9]) ? "0" : splits[9].ToString()),
                        Cabin = splits[10],
                        Embarked = splits[11]
                    });
                }
            }
            */

            IEnumerable<TitanicPrediction> predicts = model.Predict(testThis);

            Console.WriteLine("Classification Predictions");

            IEnumerable<(PassengerData sentiment, TitanicPrediction prediction)> sentimentsAndPredictions =
                testThis.Zip(predicts, (sentiment, prediction) => (sentiment, prediction));

            using (System.IO.StreamWriter file = new System.IO.StreamWriter(ResultsDataPath))
            {
                file.WriteLine("PassengerId,Survived");

                foreach (var item in sentimentsAndPredictions)
                {
                    string passengerId = item.sentiment.PassengerId.ToString();
                    int didTheySurvive = Boolean.Parse(item.prediction.IsSurvived.ToString()) ? 1 : 0;

                    file.WriteLine($"{passengerId},{didTheySurvive.ToString()}");
                }
            }

            Console.WriteLine("PassengerId,Survived");

            foreach (var item in sentimentsAndPredictions)
            {
                string passengerId = item.sentiment.PassengerId.ToString();
                string gender = item.sentiment.Sex;
                string fare = item.sentiment.Fare.ToString();
                string pclass = item.sentiment.Pclass.ToString();
                string isSurvived = item.prediction.IsSurvived.ToString();
                int didTheySurvive = Boolean.Parse(item.prediction.IsSurvived.ToString()) ? 1 : 0;

                Console.WriteLine($"Prediction: {passengerId} | Gender: {gender} | Fare: {fare} | PClass: {pclass} | Survived: '{isSurvived}'");
            }

            Console.ReadLine();
        }


        private static List<PassengerData> ReadPassengerData(string fileToRead)
        {
            //Fix the data here:
            //Use a method so that other data can be fixed too.
            TextReader reader = File.OpenText(fileToRead);

            var csv = new CsvReader(reader);
            var listOfObjects = new List<PassengerData>();

            csv.Read();
            csv.ReadHeader();

            while (csv.Read())
            {
                listOfObjects.Add(new PassengerData
                {
                    PassengerId = (float)Convert.ToDouble(csv[0].ToString()),
                    Pclass = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(csv[1]) ? "0" : csv[1].ToString()),
                    Sex = csv[3],
                    Age = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(csv[4]) ? "0" : csv[4].ToString()),
                    SibSp = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(csv[5]) ? "0" : csv[5].ToString()),
                    Parch = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(csv[6]) ? "0" : csv[6].ToString()),
                    Ticket = csv[7],
                    Fare = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(csv[8]) ? "0" : csv[8].ToString()),
                    Cabin = csv[9],
                    Embarked = csv[10]
                });
            }

            return listOfObjects;
        }

        private static PredictionModel<PassengerData, TitanicPrediction> Train_FastTreeBinaryClassifier()
        {
            var textLoader = new TextLoader(TrainDataPath).CreateFrom<PassengerData>(useHeader: true, separator: ',');

            var pipeline = new LearningPipeline()
            {
                textLoader,

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
