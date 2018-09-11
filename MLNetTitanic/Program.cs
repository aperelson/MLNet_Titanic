using System;
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
            PredictionModel<PassengerData, TitanicPrediction> model_FastTreeBinaryClassifier = Train_FastTreeBinaryClassifier();
            //PredictionModel<PassengerData, TitanicPrediction> model_FastForestBinaryClassifier = Train_FastForestBinaryClassifier();

            var testDataLoader = new TextLoader(TestDataPath).CreateFrom<PassengerData>(useHeader: true, separator: ',');
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model_FastTreeBinaryClassifier, testDataLoader);

            Console.WriteLine("");
            Console.WriteLine($"Train_FastTreeBinaryClassifier");
            Console.WriteLine($"Accuracy: {metrics.Accuracy} F1 Score: {metrics.F1Score}");
            Console.WriteLine($"True Positive: {metrics.ConfusionMatrix[0, 0]} False Positive: {metrics.ConfusionMatrix[0, 1]}");
            Console.WriteLine($"False Negative: {metrics.ConfusionMatrix[1, 0]} True Negative: {metrics.ConfusionMatrix[1, 1]}");
            Console.ReadLine();

            /*
            evaluator = new BinaryClassificationEvaluator();
            metrics = evaluator.Evaluate(model_FastForestBinaryClassifier, tempTextLoader);

            Console.WriteLine("");
            Console.WriteLine($"Train_FastForestBinaryClassifier");
            Console.WriteLine($"Accuracy: {metrics.Accuracy} F1 Score: {metrics.F1Score}");
            Console.WriteLine($"True Positive: {metrics.ConfusionMatrix[0, 0]} False Positive: {metrics.ConfusionMatrix[0, 1]}");
            Console.WriteLine($"False Negative: {metrics.ConfusionMatrix[1, 0]} True Negative: {metrics.ConfusionMatrix[1, 1]}");
            Console.ReadLine();
            */
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
                new TextLoader(TrainDataPath).CreateFrom<PassengerData>(separator: ','),

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
    }
}
