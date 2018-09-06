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
        const string DataPath = "train_data.csv";

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
            PredictionModel<PassengerData, TitanicPrediction> model2 = Train();

            var tempTextLoader = new TextLoader(DataPath).CreateFrom<PassengerData>(useHeader: true, separator: ',');
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model2, tempTextLoader);

            Console.WriteLine($"Accuracy: {metrics.Accuracy} F1 Score: {metrics.F1Score}");
            Console.WriteLine($"True Positive: {metrics.ConfusionMatrix[0, 0]} False Positive: {metrics.ConfusionMatrix[0, 1]}");
            Console.WriteLine($"False Negative: {metrics.ConfusionMatrix[1, 0]} True Negative: {metrics.ConfusionMatrix[1, 1]}");
            Console.ReadLine();
        }

        private static PredictionModel<PassengerData, TitanicPrediction> Train()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(DataPath).CreateFrom<PassengerData>(separator: ','),

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
    }
}
