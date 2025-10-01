package com.mapreduce.wc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.collections.map.HashedMap;
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

        Job j = Job.getInstance(c, "list top descriptions");

        j.setJarByClass(WordCount.class);

        j.setMapperClass(MapForListTopDescriptions.class);

        j.setReducerClass(ReduceForListTopDescriptions.class);

        j.setOutputKeyClass(Text.class);

        j.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(j, input);

        FileOutputFormat.setOutputPath(j, output);

        System.exit(j.waitForCompletion(true) ? 0 : 1);
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
            List<String[]> normalizedColumns = WordCount.normalizer.normalize(wordText);

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

        public void reduce(Text word, Iterable<IntWritable> values, Context con) throws IOException, InterruptedException {
            int sum = 0;

            for (IntWritable value : values) {
                sum += value.get();
            }

            if (sum > mostFrequentWordCount) {
                mostFrequentWordCount = sum;
                mostFrequentWord = word.toString();
            } else if (sum == mostFrequentWordCount) {
                mostFrequentWord = mostFrequentWord.concat(", ").concat(word.toString());
            } else if (sum == lessFrequentWordCount) {
                lessFrequentWord = lessFrequentWord.concat(", ").concat(word.toString());
            } else if (sum < lessFrequentWordCount) {
                lessFrequentWordCount = sum;
                lessFrequentWord = word.toString();
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            context.write(new Text("Most Frequent Word(s)"), new IntWritable(mostFrequentWordCount));
            context.write(new Text(mostFrequentWord), new IntWritable(mostFrequentWordCount));

            context.write(new Text("Least Frequent Word(s)"), new IntWritable(lessFrequentWordCount));
            context.write(new Text(lessFrequentWord), new IntWritable(lessFrequentWordCount));
        }
    }
}
