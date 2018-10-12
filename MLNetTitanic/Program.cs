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
        const string CompleteTrainDataPath = "complete_train.csv";
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

            [Column(ordinal: "3")]
            public string Name;

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
            //75%:
            //PredictionModel<PassengerData, TitanicPrediction> model = Train_FastTreeBinaryClassifier();

            //79%:
            PredictionModel<PassengerData, TitanicPrediction> model = Train_FastForestBinaryClassifier();

            //75%:
            //PredictionModel<PassengerData, TitanicPrediction> model = Train_LogisticRegressionClassifier();
            

            //var testData = CollectionDataSource.Create(ReadPassengerData(TestDataPath, true));

            //var evaluator = new BinaryClassificationEvaluator();
            //BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            //Console.WriteLine("");
            //Console.WriteLine($"Train_FastTreeBinaryClassifier");
            //Console.WriteLine($"Accuracy: {metrics.Accuracy} F1 Score: {metrics.F1Score}");
            //Console.WriteLine($"True Positive: {metrics.ConfusionMatrix[0, 0]} False Positive: {metrics.ConfusionMatrix[0, 1]}");
            //Console.WriteLine($"False Negative: {metrics.ConfusionMatrix[1, 0]} True Negative: {metrics.ConfusionMatrix[1, 1]}");

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


            var testThis = ReadPassengerData(RealTestDataPath, false);
            var predicts = model.Predict(testThis);

            Console.WriteLine("Classification Predictions");

            IEnumerable<(PassengerData sentiment, TitanicPrediction prediction)> sentimentsAndPredictions =
                testThis.Zip(predicts, (sentiment, prediction) => (sentiment, prediction));

            using (var file = new StreamWriter(ResultsDataPath))
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


        private static List<PassengerData> ReadPassengerData(string fileToRead, bool isTrainData)
        {
            //Fix the data here:
            //Use a method so that other data can be fixed too.
            TextReader reader = File.OpenText(fileToRead);

            var csv = new CsvReader(reader);
            var listOfObjects = new List<PassengerData>();
            var addAColumn = isTrainData ? 1 : 0;

            csv.Read();
            csv.ReadHeader();

            while (csv.Read())
            {
                listOfObjects.Add(new PassengerData
                {
                    PassengerId = (float)Convert.ToDouble(csv[0].ToString()),
                    Pclass = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(csv[1 + addAColumn]) ? "0" : csv[1+addAColumn].ToString()),
                    Name = csv[2 + addAColumn],
                    Sex = csv[3 + addAColumn],
                    Age = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(csv[4 + addAColumn]) ? "0" : csv[4 + addAColumn].ToString()),
                    SibSp = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(csv[5 + addAColumn]) ? "0" : csv[5 + addAColumn].ToString()),
                    Parch = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(csv[6 + addAColumn]) ? "0" : csv[6 + addAColumn].ToString()),
                    Ticket = csv[7 + addAColumn],
                    Fare = (float)Convert.ToDecimal(string.IsNullOrWhiteSpace(csv[8 + addAColumn]) ? "0" : csv[8 + addAColumn].ToString()),
                    Cabin = csv[9 + addAColumn],
                    Embarked = csv[10 + addAColumn],
                    Survived = (isTrainData && csv[1] == "1") 
                });
            }


            //Set up the age.
            //If the age is blank, then
            //if it's a 'Master' then the age is the average of the ages under 18
            //otherwise the age is the average of the ages

            var ageAboveZero = listOfObjects.Where(l => l.Age > 0);
            var ageYouths = listOfObjects.Where(l => l.Age > 0 && l.Age <= 18);
            var averageYouthAge = ageYouths.Average(l => l.Age);
            var averageAge = ageAboveZero.Average(l => l.Age);

            listOfObjects.Where(l => l.Age == 0 && l.Name.ToUpper().Contains("MASTER")).ToList().ForEach(l => l.Age = averageYouthAge);
            listOfObjects.Where(l => l.Age == 0 && !l.Name.ToUpper().Contains("MASTER")).ToList().ForEach(l => l.Age = averageAge);


            //Set up the Fares:
            var temp1 = listOfObjects.Where(l => l.Ticket == "19972");

            //Group by people in the same class boarding at the same port:
            //Multiple tickets means the fare was shared...
            //For those with multiple tickets, split the fare across
            var computedFares = listOfObjects
                .Where(l => l.Fare != 0)
                .GroupBy(p => new { p.Pclass, p.Ticket, p.Embarked, p.Fare })
                .Select(group => new {
                    Pclass = group.Key.Pclass,
                    Ticket = group.Key.Ticket,
                    Embarked = group.Key.Embarked,
                    Fare = group.Average(g2 => g2.Fare)
                })
                .GroupBy(g => new { g.Pclass, g.Embarked })
                .Select(h => new { Average = h.Average(i => i.Fare), Pclass = h.Key.Pclass, Embarked = h.Key.Embarked });

            listOfObjects.Where(l => l.Fare == 0).ToList().ForEach(l => l.Fare = computedFares.FirstOrDefault(c => c.Pclass == l.Pclass && c.Embarked == l.Embarked).Average);

            var temp2 = listOfObjects.Where(l => l.Ticket == "19972");

            return listOfObjects;
        }

        private static PredictionModel<PassengerData, TitanicPrediction> Train_FastTreeBinaryClassifier()
        {
            var dataCollection = CollectionDataSource.Create(ReadPassengerData(CompleteTrainDataPath, true));

            var pipeline = new LearningPipeline()
            {
                dataCollection,

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
            var dataCollection = CollectionDataSource.Create(ReadPassengerData(TrainDataPath, true));

            var pipeline = new LearningPipeline
            {
                dataCollection,

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
            var dataCollection = CollectionDataSource.Create(ReadPassengerData(TrainDataPath, true));

            var pipeline = new LearningPipeline
            {
                dataCollection,

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
