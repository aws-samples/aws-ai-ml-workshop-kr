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


-- 특정 조건을 가진 단일 테이블 쿼리
SELECT * FROM Artist WHERE Name LIKE 'A%';
SELECT * FROM Album WHERE Title LIKE '%Live%';
SELECT * FROM Track WHERE Composer IS NULL;
SELECT * FROM Customer WHERE Country = 'USA';
SELECT * FROM Customer WHERE Country = 'Brazil';
SELECT * FROM Employee WHERE Title = 'Sales Support Agent';
SELECT * FROM Invoice WHERE Total > 10;
SELECT * FROM Track WHERE Milliseconds > 500000;
SELECT * FROM InvoiceLine WHERE Quantity > 1;
SELECT * FROM Customer WHERE Company IS NOT NULL;

-- 정렬 및 제한 쿼리
SELECT * FROM Artist ORDER BY Name ASC LIMIT 10;
SELECT * FROM Album ORDER BY Title DESC;
SELECT * FROM Track ORDER BY Milliseconds DESC LIMIT 10;
SELECT * FROM Invoice ORDER BY InvoiceDate ASC;
SELECT * FROM Customer ORDER BY Country, City;
SELECT * FROM InvoiceLine ORDER BY UnitPrice DESC LIMIT 20;
SELECT * FROM Track ORDER BY Bytes DESC LIMIT 5;
SELECT * FROM Employee ORDER BY HireDate DESC;
SELECT DISTINCT Country FROM Customer ORDER BY Country;
SELECT DISTINCT BillingCountry FROM Invoice ORDER BY BillingCountry;

-- 집계 함수 쿼리
SELECT COUNT(*) FROM Track;
SELECT COUNT(*) FROM Album;
SELECT COUNT(*) FROM Customer;
SELECT SUM(Total) FROM Invoice;
SELECT AVG(Milliseconds) FROM Track;
SELECT MAX(Total) FROM Invoice;
SELECT MIN(UnitPrice) FROM Track;
SELECT COUNT(DISTINCT Country) FROM Customer;
SELECT COUNT(DISTINCT BillingCountry) FROM Invoice;
SELECT SUM(Bytes) / 1024 / 1024 AS TotalMegabytes FROM Track;
-- 그룹화 쿼리
SELECT AlbumId, COUNT(*) AS TrackCount FROM Track GROUP BY AlbumId;
SELECT AlbumId, SUM(Milliseconds) / 60000 AS DurationMinutes FROM Track GROUP BY AlbumId;
SELECT Country, COUNT(*) AS CustomerCount FROM Customer GROUP BY Country ORDER BY CustomerCount DESC;
SELECT GenreId, AVG(Milliseconds) / 1000 AS AvgDurationSeconds FROM Track GROUP BY GenreId;
SELECT InvoiceId, SUM(UnitPrice * Quantity) AS TotalAmount FROM InvoiceLine GROUP BY InvoiceId;
SELECT CustomerId, COUNT(*) AS InvoiceCount FROM Invoice GROUP BY CustomerId;
SELECT BillingCountry, SUM(Total) AS TotalSales FROM Invoice GROUP BY BillingCountry ORDER BY TotalSales DESC;
SELECT BillingCity, COUNT(*) AS InvoiceCount FROM Invoice GROUP BY BillingCity ORDER BY InvoiceCount DESC;
SELECT MediaTypeId, COUNT(*) AS TrackCount FROM Track GROUP BY MediaTypeId;
SELECT CustomerId, COUNT(*) AS InvoiceCount, SUM(Total) AS TotalSpent FROM Invoice GROUP BY CustomerId ORDER BY TotalSpent DESC LIMIT 10;
-- JOIN 쿼리
SELECT a.Title, ar.Name AS Artist FROM Album a JOIN Artist ar ON a.ArtistId = ar.ArtistId;
SELECT t.Name AS TrackName, a.Title AS AlbumTitle FROM Track t JOIN Album a ON t.AlbumId = a.AlbumId;
SELECT t.Name AS TrackName, g.Name AS GenreName FROM Track t JOIN Genre g ON t.GenreId = g.GenreId;
SELECT c.FirstName, c.LastName, i.InvoiceId, i.Total FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId;
SELECT i.InvoiceId, i.InvoiceDate, c.FirstName, c.LastName FROM Invoice i JOIN Customer c ON i.CustomerId = c.CustomerId;
SELECT t.Name AS TrackName, mt.Name AS MediaTypeName FROM Track t JOIN MediaType mt ON t.MediaTypeId = mt.MediaTypeId;
SELECT il.InvoiceLineId, t.Name AS TrackName FROM InvoiceLine il JOIN Track t ON il.TrackId = t.TrackId;
SELECT e.FirstName, e.LastName, c.FirstName AS CustomerFirstName, c.LastName AS CustomerLastName FROM Employee e JOIN Customer c ON e.EmployeeId = c.SupportRepId;
SELECT p.Name AS PlaylistName, t.Name AS TrackName FROM Playlist p JOIN PlaylistTrack pt ON p.PlaylistId = pt.PlaylistId JOIN Track t ON pt.TrackId = t.TrackId;
SELECT i.InvoiceId, c.Email, c.FirstName, c.LastName FROM Invoice i JOIN Customer c ON i.CustomerId = c.CustomerId WHERE i.Total > 10;
-- 서브쿼리
SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'Iron Maiden');
SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Jazz');
SELECT * FROM Track WHERE AlbumId IN (SELECT AlbumId FROM Album WHERE Title LIKE '%Greatest%');
SELECT * FROM Invoice WHERE CustomerId IN (SELECT CustomerId FROM Customer WHERE Country = 'Germany');
SELECT * FROM Employee WHERE ReportsTo = (SELECT EmployeeId FROM Employee WHERE LastName = 'Adams' AND FirstName = 'Andrew');
SELECT * FROM Track WHERE MediaTypeId = (SELECT MediaTypeId FROM MediaType WHERE Name = 'Protected AAC audio file');
SELECT * FROM InvoiceLine WHERE InvoiceId IN (SELECT InvoiceId FROM Invoice WHERE Total > 20);
SELECT * FROM Track WHERE Milliseconds > (SELECT AVG(Milliseconds) FROM Track);
SELECT * FROM Invoice WHERE Total > (SELECT AVG(Total) * 2 FROM Invoice);
SELECT * FROM Customer WHERE SupportRepId IN (SELECT EmployeeId FROM Employee WHERE Title = 'Sales Support Agent');

-- 복잡한 쿼리
SELECT ar.Name AS ArtistName, COUNT(t.TrackId) AS TrackCount 
FROM Artist ar 
JOIN Album al ON ar.ArtistId = al.ArtistId 
JOIN Track t ON al.AlbumId = t.AlbumId 
GROUP BY ar.Name 
ORDER BY TrackCount DESC 
LIMIT 10;

SELECT g.Name AS Genre, COUNT(t.TrackId) AS TrackCount, SUM(t.Milliseconds) / 60000 AS TotalMinutes 
FROM Genre g 
JOIN Track t ON g.GenreId = t.GenreId 
GROUP BY g.Name 
ORDER BY TrackCount DESC;

SELECT c.FirstName, c.LastName, SUM(i.Total) AS TotalSpent 
FROM Customer c 
JOIN Invoice i ON c.CustomerId = i.CustomerId 
GROUP BY c.CustomerId 
ORDER BY TotalSpent DESC 
LIMIT 5;

SELECT ar.Name AS Artist, a.Title AS Album, COUNT(t.TrackId) AS TrackCount 
FROM Artist ar 
JOIN Album a ON ar.ArtistId = a.ArtistId 
JOIN Track t ON a.AlbumId = t.AlbumId 
GROUP BY a.AlbumId 
ORDER BY TrackCount DESC;

SELECT c.Country, COUNT(i.InvoiceId) AS Purchases, SUM(i.Total) AS TotalRevenue 
FROM Customer c 
JOIN Invoice i ON c.CustomerId = i.CustomerId 
GROUP BY c.Country 
ORDER BY TotalRevenue DESC;

SELECT STRFTIME('%Y', InvoiceDate) AS Year, STRFTIME('%m', InvoiceDate) AS Month, COUNT(*) AS InvoiceCount, SUM(Total) AS Revenue 
FROM Invoice 
GROUP BY Year, Month 
ORDER BY Year, Month;

SELECT g.Name AS Genre, COUNT(il.InvoiceLineId) AS UnitsSold, SUM(il.UnitPrice * il.Quantity) AS Revenue 
FROM Genre g 
JOIN Track t ON g.GenreId = t.GenreId 
JOIN InvoiceLine il ON t.TrackId = il.TrackId 
GROUP BY g.Name 
ORDER BY Revenue DESC;

SELECT c.Country, g.Name AS Genre, COUNT(il.InvoiceLineId) AS UnitsSold 
FROM Customer c 
JOIN Invoice i ON c.CustomerId = i.CustomerId 
JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId 
JOIN Track t ON il.TrackId = t.TrackId 
JOIN Genre g ON t.GenreId = g.GenreId 
GROUP BY c.Country, g.Name 
ORDER BY c.Country, UnitsSold DESC;

SELECT e.FirstName, e.LastName, COUNT(i.InvoiceId) AS InvoiceCount, SUM(i.Total) AS TotalSales 
FROM Employee e 
JOIN Customer c ON e.EmployeeId = c.SupportRepId 
JOIN Invoice i ON c.CustomerId = i.CustomerId 
GROUP BY e.EmployeeId 
ORDER BY TotalSales DESC;

SELECT c.Email, COUNT(i.InvoiceId) AS Purchases, SUM(i.Total) AS TotalSpent, AVG(i.Total) AS AverageSpent 
FROM Customer c 
JOIN Invoice i ON c.CustomerId = i.CustomerId 
GROUP BY c.Email 
HAVING Purchases > 5 
ORDER BY TotalSpent DESC;

-- HAVING 절을 사용한 쿼리
SELECT Country, COUNT(*) AS CustomerCount FROM Customer GROUP BY Country HAVING CustomerCount > 5;
SELECT AlbumId, COUNT(*) AS TrackCount FROM Track GROUP BY AlbumId HAVING TrackCount > 10;
SELECT CustomerId, SUM(Total) AS TotalSpent FROM Invoice GROUP BY CustomerId HAVING TotalSpent > 40;
SELECT BillingCountry, AVG(Total) AS AvgTotal FROM Invoice GROUP BY BillingCountry HAVING AvgTotal > 6;
SELECT InvoiceId, SUM(UnitPrice * Quantity) AS LineTotal FROM InvoiceLine GROUP BY InvoiceId HAVING LineTotal > 10;

-- 특수 기능 쿼리
SELECT DISTINCT City, Country FROM Customer ORDER BY Country, City;
SELECT COUNT(*) AS TrackCount, MIN(Milliseconds) AS ShortestTrack, MAX(Milliseconds) AS LongestTrack, AVG(Milliseconds) AS AvgTrackLength FROM Track;
SELECT c.LastName || ', ' || c.FirstName AS FullName, c.Country, SUM(i.Total) AS TotalSpent FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId;
SELECT CAST((SUM(Milliseconds) / 1000 / 60 / 60) AS INTEGER) AS TotalHours FROM Track;
SELECT t.Name, t.Milliseconds / 1000 AS Seconds, CASE WHEN t.Milliseconds > 300000 THEN 'Long' WHEN t.Milliseconds < 180000 THEN 'Short' ELSE 'Medium' END AS Duration FROM Track t ORDER BY t.Milliseconds DESC LIMIT 10;