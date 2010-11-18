/*
 * Copyright (c) 2010, Marek Schmidt
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * - Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice, 
 *   this list of conditions and the following disclaimer in the documentation 
 *   and/or other materials provided with the distribution.
 * - Neither the name of the Brno University of Technology nor the names of its
 *   contributors may be used to endorse or promote products derived from this 
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * Contributor(s):
 *     Marek Schmidt <fregaham@gmail.com>
 * 
 */

package cz.vutbr.fit.nlp.lc.tools;

import cz.vutbt.fit.nlp.lc.*;

import java.io.Serializable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Random;
import java.util.Map.Entry;

import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.lucene.analysis.Token;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**

    An experiment with Reuters dataset.

    @see http://www.daviddlewis.com/resources/testcollections/reuters21578/

    @author Marek Schmidt
*/
class ReutersExperiment {

    public static class Result {
        public Result(int docid, String title, double score) {
            this.docid = docid;
            this.title = title;
            this.score = score;
        }
        int docid;
        String title;
        double score;
    }

    public static class Eval {
        double p;
        double r;
        double f1;
    }

    public static int title2docid(IndexReader reader, String title) throws Exception {
        Term term = new Term("id", title);
        TermDocs docs = reader.termDocs(term);
        docs.next();
        int docid = docs.doc();
        return docid;
    }

    public static NaiveBayesClassifier train(IndexReader reader, Set<String> pos, Set<String> neg) throws Exception {
        Map<Integer, Set<Integer>> class2ids = new HashMap<Integer, Set<Integer>>();

        class2ids.put(0, new HashSet<Integer>());
        class2ids.put(1, new HashSet<Integer>());

        for (String s : pos) {
            class2ids.get(0).add(title2docid(reader, s));
        }        

       for (String s : neg) {
            class2ids.get(1).add(title2docid(reader, s));
        }

        NaiveBayesClassifier nbc = LuceneClassification.learn(reader, class2ids, "lemmas");
        return nbc;
    }
   
    public static List<Result> search(IndexReader reader, NaiveBayesClassifier nbc) throws Exception {
        List<Result> ret = new LinkedList<Result> ();
        LuceneClassification classification = new LuceneClassification(reader, nbc, "lemmas", 0);

        LuceneClassification.Iteration iter = classification.steps(null, 50);

        // System.out.println("Iteration: " + iter.iteration);
        for (Map.Entry<Integer, Double> entry : iter.id2logscore.entrySet()) {

            Document doc = reader.document(entry.getKey());
            Field field = doc.getField("id");
            //TokenStream stream = field.tokenStreamValue();
            String id = field.stringValue();

            ret.add (new Result(entry.getKey(), id, entry.getValue()));
            //System.out.println("" + id + "\t" + Math.exp(entry.getValue()));
        }


        return ret;
    }

    public static Eval evaluate(List<Result> results, Set<String> allTestPoses) {
        double thr = 0;

        int tp = 0;
        int fp = 0;
        //int tn = 0;

        int total = 0;
        double p = 0.0;
        double r = 0.0;
        double f1 = 0.0;

        Eval e = null;

        for (Result res : results) {
            if (res.title.startsWith("test/")) {

                total += 1;

                // System.out.println("" + Math.exp(res.score));
                //if (res.score > thr) {
                    if (allTestPoses.contains(res.title)) {
                        tp += 1;
                    }
                    else {
                        fp += 1;
                    }
                //}

                p = (0.0 + tp) / (0.0 + total);
                r = (0.0 + tp) / (0.0 + allTestPoses.size());
                f1 = 2 * (p*r) / (p + r);

                // System.out.println("p: " + p + ", r: " + r + ", f1: " + f1);

                if (r != 0.0 && r >= p && e == null) {
                    e = new Eval();
                    e.p = p;
                    e.r = r;    
                    e.f1 = f1;
                }
            }
        }

        if(e == null) {
            p = (0.0 + tp) / (0.0 + total);
            r = (0.0 + tp) / (0.0 + allTestPoses.size());
            f1 = 2 * (p*r) / (p + r);

            e = new Eval();
            e.p = p;
            e.r = r;    
            e.f1 = f1;
        }

        return e;
    }

    public static void main(String[] args) throws Exception {

        Random random = new Random();

        IndexReader reader;
        
        Directory dir = FSDirectory.getDirectory(new File(args[0]));

        String[] tags = {"earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"};

        reader = IndexReader.open(dir);

        for (String tag : tags) {
            Term term = new Term("tag", tag);
        
            Set<String> allTrainPoses = new HashSet<String>();
            Set<String> allTestPoses = new HashSet<String>();
            TermDocs docs = reader.termDocs(term);
            while(docs.next()) {
                int docid = docs.doc();
                Document doc = reader.document(docid);
                Field field = doc.getField("id");
                String title = field.stringValue();
    
                if (title.startsWith("training/")) {
                    allTrainPoses.add (title);
                }
                else if (title.startsWith("test/")) {
                    allTestPoses.add (title);
                }
            }

            Set<String> trainPoses = new HashSet<String>();
            Set<String> trainNeges = new HashSet<String>();

            // randomly choose first example
            //trainPoses.add(new ArrayList<String>(allTrainPoses).get(random.nextInt(allTrainPoses.size())));
            //trainPoses.add(new ArrayList<String>(allTrainPoses).get(0));

            // put all the training data in there...
            trainPoses.addAll(allTrainPoses);

            for (int i = 0; i < reader.numDocs(); ++i) {
                Document doc = reader.document(i);
                Field field = doc.getField("id");
                String title = field.stringValue();
                if (title.startsWith("training/") && !trainPoses.contains(title)) {
                    trainNeges.add(title);
                }
            }

            boolean end = false;
            while (!end) {
                end = true;

                NaiveBayesClassifier nbc = train(reader, trainPoses, trainNeges);
                List<Result> results = search(reader, nbc);

                Collections.sort(results, new Comparator<Result>() {
                    public int compare(Result r1, Result r2) {
                        return Double.compare(r2.score, r1.score);
                    }
                });

                List<Result> trainResults = new LinkedList<Result>();
    
                Eval e = evaluate(results, allTestPoses);

                System.out.println("" + tag + " " + (trainPoses.size() + trainNeges.size()) + " " + trainPoses.size() + " " + trainNeges.size() + " " + e.p + " " + e.r + " " + e.f1);

                for (Result res : results) {
                    if (trainPoses.contains(res.title)) {
                //        System.out.println("" + res.title + " score: " + res.score);
                    }

                    if (trainPoses.contains(res.title) || trainNeges.contains(res.title)) {
                        continue;
                    }
                    if (res.title.startsWith("training/")) {
                        trainResults.add (res);
                    }
                }

                // strategy to choose the best next training example... we choose closest to zero...
                Collections.sort(trainResults, new Comparator<Result>() {
                    public int compare(Result r1, Result r2) {
                        // return Double.compare(Math.abs(r1.score), Math.abs(r2.score));
                        return - Double.compare(r1.score, r2.score);
                    }
                });

                if (trainResults.size() > 0) {
                    Result selected = trainResults.get(0);
                    if (allTrainPoses.contains(selected.title)) {
                        trainPoses.add (selected.title);
                    }
                    else {
                        trainNeges.add (selected.title);
                    }

                    end = false;
                }
            }
        }
    }
}
