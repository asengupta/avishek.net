<?php
/*
Template Name: Search
*/
?>

<?php get_header(); ?>

		<!-- Maincontent start -->
		<div id="maincontent">
			<h2>Search results for <?php echo $s; ?></h2>

<?php if (have_posts()) : ?>
			<p>Your search for <em><?php echo $s; ?></em> yielded <?php $NumResults = $wp_query->found_posts; echo $NumResults; ?> results.</p><hr />
<?php while (have_posts()) : the_post(); ?>
			<div class="post">
				<h3><a href="<?php the_permalink() ?>" rel="bookmark" title="Permanent lenke til: <?php the_title_attribute(); ?>"><?php the_title(); ?></a></h3>
				<div class="date"><?php the_date(); ?> - <?php the_time(); ?></div>
				<?php the_excerpt(''); ?>
			</div>
<?php endwhile; ?>

<?php else : ?>
			<p>Your search for <em><?php echo $s; ?></em> yielded no results.</p>
			<?php include (TEMPLATEPATH . '/searchform.php'); ?>
<?php endif; ?>

		</div>
		<!-- Maincontent end -->

<?php get_sidebar(); ?>

<?php get_footer(); ?>

