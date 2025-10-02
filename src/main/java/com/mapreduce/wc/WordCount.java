package com.mapreduce.wc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.conf.Configuration;

import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.IntWritable;

import org.apache.hadoop.io.LongWritable;

import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;

import org.apache.hadoop.mapreduce.Mapper;

import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;

import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.util.GenericOptionsParser;

public class WordCount {
    static Normalizer normalizer = new Normalizer();

    public static void main(String[] args) throws Exception {
        Configuration c = new Configuration();
        String[] files = new GenericOptionsParser(c, args).getRemainingArgs();

        if (files.length < 2) {
            System.err.println("Usage: WordCount <input path> <output path>");
            System.exit(-1);
        }

        Path input = new Path(files[0]);
        Path output = new Path(files[1]);
        Path output2 = new Path(files[1] + "_2");

        Job j = Job.getInstance(c, "list top descriptions");
        j.setJarByClass(WordCount.class);
        j.setMapperClass(MapForListTopDescriptions.class);
        j.setReducerClass(ReduceForListTopDescriptions.class);
        j.setOutputKeyClass(Text.class);
        j.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(j, input);
        FileOutputFormat.setOutputPath(j, output);

        Job j2 = Job.getInstance(c, "list title with longest and shortest description");
        j2.setJarByClass(WordCount.class);
        j2.setMapperClass(MapForTitle.class);
        j2.setReducerClass(ReduceForTitle.class);
        j2.setOutputKeyClass(Text.class);
        j2.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(j2, input);
        FileOutputFormat.setOutputPath(j2, output2);

        boolean success = j.waitForCompletion(true);
        if (success) {
            success = j2.waitForCompletion(true);
        }

        System.exit(success ? 0 : 1);
    }

    public static class Normalizer {
        public List<String[]> normalize(Text value) throws IOException, InterruptedException {
            String text = value.toString().trim();

            String[] lines = text.split("\\r?\\n");

            ArrayList<String[]> normalizedColumns = new ArrayList<String[]>();

            for (String line : lines) {
                String[] columns = line.split(",");
                String[] newColumns = new String[2];

                String title = columns[2].toLowerCase().replaceAll("[^a-zA-Z0-9\\s]", " ");
                String description = columns[11].toLowerCase().replaceAll("[^a-zA-Z0-9\\s]", " ");

                newColumns[0] = title;
                newColumns[1] = description;

                normalizedColumns.add(newColumns);
            }

            return normalizedColumns;
        }
    }

    public static class MapForListTopDescriptions extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);

        private final Text wordText = new Text();

        public void map(LongWritable key, Text value, Context con) throws IOException, InterruptedException {
            List<String[]> normalizedColumns = WordCount.normalizer.normalize(value);

            for (String[] normalizedColumn : normalizedColumns) {
                String[] words = normalizedColumn[1].split("\\s+");
                for (String word : words) {
                    if (!word.isEmpty()) {
                        wordText.set(word);
                        con.write(wordText, one);
                    }
                }
            }
        }
    }

    public static class ReduceForListTopDescriptions extends Reducer<Text, IntWritable, Text, IntWritable> {
        String mostFrequentWord = null;
        int mostFrequentWordCount = 0;
        String lessFrequentWord = null;
        int lessFrequentWordCount = Integer.MAX_VALUE;
        int totalSum = 0;

        public void reduce(Text word, Iterable<IntWritable> values, Context con) throws IOException, InterruptedException {
            int sum = 0;

            for (IntWritable value : values) {
                sum += value.get();
            }

            totalSum += sum;

            if (sum > mostFrequentWordCount) {
                mostFrequentWordCount = sum;
                mostFrequentWord = word.toString();
            } 
            if (sum == mostFrequentWordCount) {
                mostFrequentWord = mostFrequentWord.concat(", ").concat(word.toString());
            } 
            if (sum < lessFrequentWordCount) {
                lessFrequentWordCount = sum;
                lessFrequentWord = word.toString();
            }
            if (sum == lessFrequentWordCount) {
                lessFrequentWord = lessFrequentWord.concat(", ").concat(word.toString());
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            context.write(new Text("Most Frequent Word(s)"), new IntWritable(mostFrequentWordCount));
            context.write(new Text(mostFrequentWord), new IntWritable(mostFrequentWordCount));

            context.write(new Text("Least Frequent Word(s)"), new IntWritable(lessFrequentWordCount));
            context.write(new Text(lessFrequentWord), new IntWritable(lessFrequentWordCount));

            context.write(new Text("Total Word Count"), new IntWritable(totalSum));
        }
    }

    public class TitleWithDescription {
        String title;
        String description;

        public TitleWithDescription(String title, String description) {
            this.title = title;
            this.description = description;
        }

        public String getTitle() {
            return title;
        }

        public String getDescription() {
            return description;
        }
    }

    public static class MapForTitle extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final Text wordText = new Text();

        public void map(LongWritable key, Text value, Context con) throws IOException, InterruptedException {
            List<String[]> normalizedColumns = WordCount.normalizer.normalize(value);

            for (String[] normalizedColumn : normalizedColumns) {
                wordText.set(normalizedColumn[0]);
                String[] words = normalizedColumn[1].split("\\s+");
                String[] cleanedWords = Arrays.stream(words).filter(word -> !word.isEmpty()).toArray(String[]::new);

                con.write(wordText, new IntWritable(cleanedWords.length));
            }
        }
    }

    public static class ReduceForTitle extends Reducer<Text, IntWritable, Text, IntWritable> {
        String longestDescriptionTitle = "";
        int longestDescriptionTitleLength = 0;
        String shortestDescriptionTitle = "";
        int shortestDescriptionTitleLength = Integer.MAX_VALUE;

        public void reduce(Text word, Iterable<IntWritable> values, Context con) throws IOException, InterruptedException {
            int sum = values.iterator().next().get();

            if (sum > longestDescriptionTitleLength) {
                longestDescriptionTitleLength = sum;
                longestDescriptionTitle = word.toString();
            } 
            if (sum == longestDescriptionTitleLength) {
                longestDescriptionTitle = longestDescriptionTitle.concat(", ").concat(word.toString());
            } 
            if (sum < shortestDescriptionTitleLength) {
                shortestDescriptionTitleLength = sum;
                shortestDescriptionTitle = word.toString();
            }
            if (sum == shortestDescriptionTitleLength) {
                shortestDescriptionTitle = shortestDescriptionTitle.concat(", ").concat(word.toString());
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            context.write(new Text("Longest Description Title(s)"), new IntWritable(longestDescriptionTitleLength));
            context.write(new Text(longestDescriptionTitle), new IntWritable(longestDescriptionTitleLength));

            context.write(new Text("Shortest Description Title(s)"), new IntWritable(shortestDescriptionTitleLength));
            context.write(new Text(shortestDescriptionTitle), new IntWritable(shortestDescriptionTitleLength));
        }
    }
}
