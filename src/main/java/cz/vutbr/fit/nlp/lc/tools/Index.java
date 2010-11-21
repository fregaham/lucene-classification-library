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


import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 *  Helpful tool to index data. The argument is the path to the index it should create.
 *  The data format is a tab delimented text file
 *  {@code id \t <title> \t <space delimeted bag of features, such as plain text> \t <space delimeted `tags', or classes, optional> }
 *
 *  {@code java -cp lucene-core-2.9.1.jar:. cz.vutbr.fit.nlp.lc.tools.Index index}
 *   @author Marek Schmidt
 */
class Index {

    public static void main(String[] args) throws Exception {

        IndexWriter writer;
        
        Directory dir = FSDirectory.getDirectory(new File(args[0]));
        IndexWriter.unlock(dir);
                
        writer = new IndexWriter(dir, new WhitespaceAnalyzer(), true, IndexWriter.MaxFieldLength.LIMITED);

        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        String line;
        long i = 0;
        while( (line = in.readLine()) != null) {

            i++;

            String[] split = line.split("\t", 4);

            String id = null;
            String title = null;
            String lemmas = null;
            String tags = null;

            if (split.length == 3 || split.length == 4) {
                id = split[0];
                title = split[1];
                lemmas = split[2];
            }
            else {
                System.err.println("Wrong line " + i + " : " + line);
                continue;
            }

            if (split.length == 4) {
                tags = split[3];
            }
            
            Document doc = new Document();
            doc.add(new Field("id", id, Field.Store.YES, Field.Index.NOT_ANALYZED));
            doc.add(new Field("title", title, Field.Store.YES, Field.Index.ANALYZED));
            doc.add(new Field("lemmas", lemmas, Field.Store.YES, Field.Index.ANALYZED));

            if (tags != null) {
                doc.add(new Field("tag", tags, Field.Store.YES, Field.Index.ANALYZED));
            }

            writer.addDocument(doc);
        }

        writer.flush();
        writer.close();
    }
}
