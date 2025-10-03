package com.mapreduce.wc;

import java.io.IOException;
import java.io.StringReader;
import java.util.*;

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

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class WordCount {
    static Normalizer normalizer = new Normalizer();

    public static void main(String[] args) throws Exception {
        Configuration c = new Configuration();
        String[] files = new GenericOptionsParser(c, args).getRemainingArgs();

        if (files.length < 2) {
            System.err.println("Usage: WordCount <input path> <output path>");
            System.exit(-1);
        }

        Path input = new Path(files[1]);
        Path output = new Path(files[2]);
        Path output2 = new Path(files[2] + "_2");

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
        public List<String[]> normalize(Text value) throws IOException {
            String text = value.toString().trim();
            List<String[]> normalizedColumns = new ArrayList<>();
            String[] lines = text.split("\\r?\\n");
            CSVFormat csvFormat = CSVFormat.DEFAULT
                    .builder()
                    .setSkipHeaderRecord(true)
                    .build();

            for (String line : lines) {
                try (CSVParser parser = CSVParser.parse(line, csvFormat)) {
                    for (CSVRecord record : parser) {
                        if (record.size() > 11) {
                            String[] newColumns = new String[2];
                            String title = record.get(2)
                                    .toLowerCase()
                                    .replaceAll("[^a-zA-Z0-9\\s]", " ");
                            String description = record.get(11)
                                    .toLowerCase()
                                    .replaceAll("[^a-zA-Z0-9\\s]", " ");
                            newColumns[0] = title;
                            newColumns[1] = description;
                            normalizedColumns.add(newColumns);
                        }
                    }
                } catch (Exception e) {
                }
            }
            return normalizedColumns;
        }
    }

    public static class MapForListTopDescriptions extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);

        private final Text wordText = new Text();

        private final String[] stopWords = {
                "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
                "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
                "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
                "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
                "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
                "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
                "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
                "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
                "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
                "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
                "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
                "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
                "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
                "yourselves", "s"
        };

        public void map(LongWritable key, Text value, Context con) throws IOException, InterruptedException {
            List<String[]> normalizedColumns = WordCount.normalizer.normalize(value);

            for (String[] normalizedColumn : normalizedColumns) {
                String[] words = normalizedColumn[1].split("\\s+");
                words = Arrays.stream(words).filter(word -> !Arrays.asList(stopWords).contains(word)).toArray(String[]::new);
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
        TreeMap<Integer, List<String>> top5LessFrequentDescriptions = new TreeMap<>();
        TreeMap<Integer, List<String>> top5MostFrequentDescriptions = new TreeMap<>(Collections.reverseOrder());

        int totalSum = 0;

        public void reduce(Text word, Iterable<IntWritable> values, Context con) throws IOException, InterruptedException {
            if (word.toString().equals("description")) {
                return;
            }
            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }

            totalSum += sum;

            top5MostFrequentDescriptions.computeIfAbsent(sum, k -> new ArrayList<>()).add(word.toString());
            if (top5MostFrequentDescriptions.size() > 5) {
                top5MostFrequentDescriptions.pollLastEntry();
            }

            top5LessFrequentDescriptions.computeIfAbsent(sum, k -> new ArrayList<>()).add(word.toString());
            if (top5LessFrequentDescriptions.size() > 5) {
                top5LessFrequentDescriptions.pollLastEntry();
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            context.write(new Text("Top 5 Most Frequent Words:"), null);
            for (Map.Entry<Integer, List<String>> entry : top5MostFrequentDescriptions.entrySet()) {
                for (String word : entry.getValue()) {
                    context.write(new Text(word), new IntWritable(entry.getKey()));
                }
            }

            context.write(new Text("Top 5 Less Frequent Words:"), null);
            for (Map.Entry<Integer, List<String>> entry : top5LessFrequentDescriptions.entrySet()) {
                for (String word : entry.getValue()) {
                    context.write(new Text(word), new IntWritable(entry.getKey()));
                }
            }

            context.write(new Text("Total Word Count:"), new IntWritable(totalSum));
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
        int totalSum = 0;
        int totalTitles = 0;

        public void reduce(Text word, Iterable<IntWritable> values, Context con) throws IOException, InterruptedException {
            if (word.toString().equals("title")) {
                return;
            }
            int sum = values.iterator().next().get();

            totalSum += sum;
            totalTitles += 1;

            if (sum > longestDescriptionTitleLength) {
                longestDescriptionTitleLength = sum;
                longestDescriptionTitle = word.toString();
            } else if (sum == longestDescriptionTitleLength) {
                longestDescriptionTitle = longestDescriptionTitle.concat(", ").concat(word.toString());
            }
            if (sum < shortestDescriptionTitleLength) {
                shortestDescriptionTitleLength = sum;
                shortestDescriptionTitle = word.toString();
            } else if (sum == shortestDescriptionTitleLength) {
                shortestDescriptionTitle = shortestDescriptionTitle.concat(", ").concat(word.toString());
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            context.write(new Text("Longest Description Title(s)"), new IntWritable(longestDescriptionTitleLength));
            context.write(new Text(longestDescriptionTitle), new IntWritable(longestDescriptionTitleLength));

            context.write(new Text("Shortest Description Title(s)"), new IntWritable(shortestDescriptionTitleLength));
            context.write(new Text(shortestDescriptionTitle), new IntWritable(shortestDescriptionTitleLength));

            int averageWordsPerDescription = totalTitles == 0 ? 0 : totalSum / totalTitles;
            context.write(new Text("Average Words per Description"), new IntWritable(averageWordsPerDescription));
        }
    }
}
