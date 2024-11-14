SELECT * FROM Artist;
SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');
SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');
SELECT SUM(Milliseconds) FROM Track;
SELECT * FROM Customer WHERE Country = 'Canada';
SELECT COUNT(*) FROM Track WHERE AlbumId = 5;
SELECT COUNT(*) FROM Invoice;
SELECT * FROM Track WHERE Milliseconds > 300000;
SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;
SELECT COUNT(*) FROM Employee