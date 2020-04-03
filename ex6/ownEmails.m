function X = ownEmails (emailFile)

%Allows email to be read into X, y, and word indices
contents = readFile(emailFile);
word_indices = processEmail(contents);

X = emailFeatures(word_indices);

end