package cz.vutbr.fit.nlp.lc;

import java.util.Map;
import java.util.Set;
import java.util.HashMap;
import java.util.HashSet;

import junit.framework.Assert;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;

public class TestLuceneClassification 
    extends TestCase
{
    public TestLuceneClassification( String testName )
    {
        super( testName );
    }

    public static Test suite()
    {
        return new TestSuite( TestLuceneClassification.class );
    }

    static final String[] testDocuments = {"aaa bbb", "aaa ccc", "ccc ddd", "aaa eee", "ccc eee", "ddd eee", "fff ggg", "ddd aaa"};
    static final int[] testDocumentClasses = {1, 1, 0, 1, 0, 0, 0, 1};

    public void testSearch() throws java.io.IOException
    {
        // Open the writer
        IndexWriter writer;
        RAMDirectory dir = new RAMDirectory();
        writer = new IndexWriter( dir, new WhitespaceAnalyzer(), true, IndexWriter.MaxFieldLength.LIMITED );

        // Index the test documents
        for( int i = 0; i < testDocuments.length; ++i ) {
            Document doc = new Document();
            doc.add( new Field( "id", "" + i, Field.Store.YES, Field.Index.NOT_ANALYZED ) );
            doc.add( new Field( "lemmas", testDocuments[i], Field.Store.YES, Field.Index.ANALYZED ) );
            writer.addDocument( doc );
        }

        // close the writer
        writer.flush();
        writer.close();

        // open the reader
        IndexReader reader;        
        reader = IndexReader.open( dir );

        // we use the first n / 2 documents for training
        Map<Integer, Set<Integer>> class2ids = new HashMap<Integer, Set<Integer>>();
        class2ids.put( 0, new HashSet<Integer> () );
        class2ids.put( 1, new HashSet<Integer> () );

        for( int i = 0; i < testDocumentClasses.length / 2; ++i ) {

            // the class2ids identifies the docids, not our "id"s... 
            // we find the docid of this document

            Term term = new Term( "id", "" + i );
            TermDocs docs = reader.termDocs( term );
            docs.next();
            int docid = docs.doc();

            class2ids.get( testDocumentClasses[i] ).add( docid );
        }

        NaiveBayesClassifier nbc;
        nbc = LuceneClassification.learn( reader, class2ids, "lemmas" );

        // Try to 
        LuceneClassification classification = new LuceneClassification( reader, nbc, "lemmas", 1 );
        LuceneClassification.Iteration iter = null;
        while( classification.hasNext( iter ) ) {
            iter = classification.step(iter);
            System.err.println("Iteration: " + iter.getIteration() + ", feature: " + classification.getFeatures().get(iter.getIteration()) + " ll: " + nbc.loglikelihoods.get(1).get(classification.getFeatures().get(iter.getIteration())));
        }

        // we should have at least the training documents scored... and at
        // least one more...
        Assert.assertTrue( iter.getId2LogScore().size() > testDocumentClasses.length / 2 );

        for( Map.Entry<Integer, Double> entry : iter.getId2LogScore().entrySet() ) {
            int docid = entry.getKey();
            double score = entry.getValue();

            Document doc = reader.document( docid );
            Field field = doc.getField( "id" );
            int id = Integer.parseInt( field.stringValue() );
        
            System.err.println( "" + id + ": " + score );    
            Assert.assertTrue( (score > 0.0) == (testDocumentClasses[id] == 1) );
        }
    } 
}

